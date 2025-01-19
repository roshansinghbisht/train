"""Train the model"""

import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import utils
import model.net as net
import model.data_loader_custom as data_loader
from evaluate import evaluate
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/ocr_data_mini',
#                     help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

device = torch.device("mps")

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    train_losss = 0.0
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            # if params.cuda:
            #     train_batch, labels_batch = train_batch.cuda(
            #         non_blocking=True), labels_batch.cuda(non_blocking=True)
            #@14/03
            train_batch, labels_batch = train_batch.to("mps"), labels_batch.to("mps")
            #@
            # convert to torch Variables
            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            # compute model output and loss
            print("train_batch shape", train_batch.shape)
            print("labels_batch shape", labels_batch.shape)
            
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()
            train_losss += loss.item() 
            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    train_losss /= len(dataloader)
    return train_losss


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    train_losses = []  # To store the training losses
    val_losses = [] # To store the validation losses

    for epoch in range(params.num_epochs):
        # Run one epoch
        if epoch == 50:
            for param in model.parameters():
                param.requires_grad = True
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_losss = train(model, optimizer, loss_fn, train_dataloader, metrics, params)
        train_losses.append(train_losss)
        # Evaluate for one epoch on validation set
        val_metrics, val_losss = evaluate(model, loss_fn, val_dataloader, metrics, params)
        val_losses.append(val_losss)
        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
    return train_losses, val_losses

def plot_loss_curves(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # If save_path is provided, save the plot
    if save_path:
        plt.savefig(save_path)  # Save the figure to the given path
        print(f"Plot saved as {save_path}")

    # Show the plot
    # plt.show()


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    # params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    # if params.cuda:
    #     torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders

    dataloaders = data_loader.fetch_dataloader(
        ['train', 'val'], params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    model = models.mobilenet_v2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    print("model before: ", model)
    
    model.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                    nn.Linear(in_features=1280, out_features=640, bias=True),
                                    nn.BatchNorm1d(num_features=640),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(in_features=640, out_features=320, bias=True),
                                    nn.BatchNorm1d(num_features=320),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(in_features=320, out_features=160, bias=True),
                                    nn.BatchNorm1d(num_features=160),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(in_features=160, out_features=80, bias=True),
                                    nn.BatchNorm1d(num_features=80),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(in_features=80, out_features=40, bias=True),
                                    nn.BatchNorm1d(num_features=40),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(in_features=40, out_features=7, bias=True),
                                    )
    
    # model = models.efficientnet_b0(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False

    # model.classifier = nn.Sequential(nn.Dropout(p=0.5),
    #                             nn.Linear(in_features=1280, out_features=640, bias=True),
    #                             nn.BatchNorm1d(num_features=640),
    #                             nn.ReLU(),
    #                             nn.Dropout(p=0.5),
    #                             nn.Linear(in_features=640, out_features=320, bias=True),
    #                             nn.BatchNorm1d(num_features=320),
    #                             nn.ReLU(),
    #                             nn.Dropout(p=0.5),
    #                             nn.Linear(in_features=320, out_features=160, bias=True),
    #                             nn.BatchNorm1d(num_features=160),
    #                             nn.ReLU(),
    #                             nn.Dropout(p=0.5),
    #                             nn.Linear(in_features=160, out_features=80, bias=True),
    #                             nn.BatchNorm1d(num_features=80),
    #                             nn.ReLU(),
    #                             nn.Dropout(p=0.5),
    #                             nn.Linear(in_features=80, out_features=40, bias=True),
    #                             nn.BatchNorm1d(num_features=40),
    #                             nn.ReLU(),
    #                             nn.Dropout(p=0.5),
    #                             nn.Linear(in_features=40, out_features=7, bias=True),
    #                             )
    
    
    print(model)
    print("///////////////////////////////")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # # # fetch loss function and metrics
    loss_fn = nn.CrossEntropyLoss()
    metrics = net.metrics

    # # # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    # train_losses, val_losses = train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
    #                    args.restore_file)
    
    # save_path = "/Users/roshanbisht/Documents/code/assignments/godigit/output/your_loss_plot.png"  # Specify where you want to save the plot
    # plot_loss_curves(train_losses, val_losses, save_path)
    
