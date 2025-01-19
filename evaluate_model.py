import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torchvision

# Function to load the saved model
def load_model(model_path, model_class):
    device = torch.device("mps")
    model_class.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model, device

# Function to create the test data loader
def get_test_loader(test_data_path, batch_size=8):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, test_dataset.classes

# Function to evaluate the model
def evaluate_model(model, test_loader, device):
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation during evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get class index with highest score

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    return accuracy, all_preds, all_labels


# Function to calculate additional metrics
def calculate_metrics(all_preds, all_labels, class_names):
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report (Precision, Recall, F1-Score)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)


# Main evaluation script
if __name__ == "__main__":
    # Define paths
    model_path = "v2_model_state_mobilenetv2.pt"  # Path to the saved model
    test_data_path = "/Users/roshanbisht/Documents/code/assignments/godigit/data_splits/test"  # Path to the test folder

    # Load model
    model = torchvision.models.mobilenet_v2(pretrained=True)
    #change the last layer the same way it was done during training.
    model.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                    nn.Linear(in_features=1280, out_features=640, bias=True),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(num_features=640),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(in_features=640, out_features=320, bias=True),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(num_features=320),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(in_features=320, out_features=196, bias=True),
                                    )

    model, device = load_model(model_path, model)

    # Load test data
    test_loader, class_names = get_test_loader(test_data_path, batch_size=16)

    # Evaluate the model
    accuracy, all_preds, all_labels = evaluate_model(model, test_loader, device)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Calculate additional metrics
    calculate_metrics(all_preds, all_labels, class_names)
