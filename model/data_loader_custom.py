import random
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.Resize((224, 224)),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomPerspective(0.1, 0.1),
    transforms.RandomRotation(degrees=(-2, 2)),
    transforms.RandomAutocontrast(0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # transform it into a torch tensor


# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize((224, 224)),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # transform it into a torch tensor


class CarClassifierDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        # self.filenames = os.listdir(data_dir)
        # self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]
        # print("self.filenames:", self.filenames)
        # self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]

        self.filenames = []
        for folder_name in os.listdir(data_dir):
            try:
                filenames = [os.path.join(data_dir, folder_name, f) for f in os.listdir(os.path.join(data_dir, folder_name)) if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
            except:
                print("folder_name: ", folder_name)
                continue
            self.filenames += filenames

        self.labels = [os.path.split(os.path.split(filename)[0])[-1] for filename in self.filenames]
        # print("self.labels: ", self.labels)
        self.transform = transform


    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        label_id = self.label2id(str(self.labels[idx]))
        return image, label_id


    def label2id(self, label):
        name_list = ['AM General Hummer SUV 2000', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012',
         'Acura TL Type-S 2008', 'Acura TSX Sedan 2012', 'Acura Integra Type R 2001',
         'Acura ZDX Hatchback 2012', 'Aston Martin V8 Vantage Convertible 2012',
         'Aston Martin V8 Vantage Coupe 2012', 'Aston Martin Virage Convertible 2012',
         'Aston Martin Virage Coupe 2012', 'Audi RS 4 Convertible 2008', 'Audi A5 Coupe 2012',
         'Audi TTS Coupe 2012', 'Audi R8 Coupe 2012', 'Audi V8 Sedan 1994', 'Audi 100 Sedan 1994',
         'Audi 100 Wagon 1994', 'Audi TT Hatchback 2011', 'Audi S6 Sedan 2011', 'Audi S5 Convertible 2012',
         'Audi S5 Coupe 2012', 'Audi S4 Sedan 2012', 'Audi S4 Sedan 2007', 'Audi TT RS Coupe 2012',
         'BMW ActiveHybrid 5 Sedan 2012', 'BMW 1 Series Convertible 2012', 'BMW 1 Series Coupe 2012',
         'BMW 3 Series Sedan 2012', 'BMW 3 Series Wagon 2012', 'BMW 6 Series Convertible 2007',
         'BMW X5 SUV 2007', 'BMW X6 SUV 2012', 'BMW M3 Coupe 2012', 'BMW M5 Sedan 2010',
         'BMW M6 Convertible 2010', 'BMW X3 SUV 2012', 'BMW Z4 Convertible 2012',
         'Bentley Continental Supersports Conv. Convertible 2012', 'Bentley Arnage Sedan 2009',
         'Bentley Mulsanne Sedan 2011', 'Bentley Continental GT Coupe 2012',
         'Bentley Continental GT Coupe 2007', 'Bentley Continental Flying Spur Sedan 2007',
         'Bugatti Veyron 16.4 Convertible 2009', 'Bugatti Veyron 16.4 Coupe 2009',
         'Buick Regal GS 2012', 'Buick Rainier SUV 2007', 'Buick Verano Sedan 2012',
         'Buick Enclave SUV 2012', 'Cadillac CTS-V Sedan 2012', 'Cadillac SRX SUV 2012',
         'Cadillac Escalade EXT Crew Cab 2007', 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',
         'Chevrolet Corvette Convertible 2012', 'Chevrolet Corvette ZR1 2012',
         'Chevrolet Corvette Ron Fellows Edition Z06 2007', 'Chevrolet Traverse SUV 2012',
         'Chevrolet Camaro Convertible 2012', 'Chevrolet HHR SS 2010', 'Chevrolet Impala Sedan 2007',
         'Chevrolet Tahoe Hybrid SUV 2012', 'Chevrolet Sonic Sedan 2012',
         'Chevrolet Express Cargo Van 2007', 'Chevrolet Avalanche Crew Cab 2012',
         'Chevrolet Cobalt SS 2010', 'Chevrolet Malibu Hybrid Sedan 2010',
         'Chevrolet TrailBlazer SS 2009', 'Chevrolet Silverado 2500HD Regular Cab 2012', 'Chevrolet Silverado 1500 Classic Extended Cab 2007', 'Chevrolet Express Van 2007', 'Chevrolet Monte Carlo Coupe 2007', 'Chevrolet Malibu Sedan 2007', 'Chevrolet Silverado 1500 Extended Cab 2012', 'Chevrolet Silverado 1500 Regular Cab 2012', 'Chrysler Aspen SUV 2009', 'Chrysler Sebring Convertible 2010', 'Chrysler Town and Country Minivan 2012', 'Chrysler 300 SRT-8 2010', 'Chrysler Crossfire Convertible 2008', 'Chrysler PT Cruiser Convertible 2008', 'Daewoo Nubira Wagon 2002', 'Dodge Caliber Wagon 2012', 'Dodge Caliber Wagon 2007', 'Dodge Caravan Minivan 1997', 'Dodge Ram Pickup 3500 Crew Cab 2010', 'Dodge Ram Pickup 3500 Quad Cab 2009', 'Dodge Sprinter Cargo Van 2009', 'Dodge Journey SUV 2012', 'Dodge Dakota Crew Cab 2010', 'Dodge Dakota Club Cab 2007', 'Dodge Magnum Wagon 2008', 'Dodge Challenger SRT8 2011', 'Dodge Durango SUV 2012', 'Dodge Durango SUV 2007', 'Dodge Charger Sedan 2012', 'Dodge Charger SRT-8 2009', 'Eagle Talon Hatchback 1998', 'FIAT 500 Abarth 2012', 'FIAT 500 Convertible 2012', 'Ferrari FF Coupe 2012', 'Ferrari California Convertible 2012', 'Ferrari 458 Italia Convertible 2012', 'Ferrari 458 Italia Coupe 2012', 'Fisker Karma Sedan 2012', 'Ford F-450 Super Duty Crew Cab 2012', 'Ford Mustang Convertible 2007', 'Ford Freestar Minivan 2007', 'Ford Expedition EL SUV 2009', 'Ford Edge SUV 2012', 'Ford Ranger SuperCab 2011', 'Ford GT Coupe 2006', 'Ford F-150 Regular Cab 2012', 'Ford F-150 Regular Cab 2007', 'Ford Focus Sedan 2007', 'Ford E-Series Wagon Van 2012', 'Ford Fiesta Sedan 2012', 'GMC Terrain SUV 2012', 'GMC Savana Van 2012', 'GMC Yukon Hybrid SUV 2012', 'GMC Acadia SUV 2012', 'GMC Canyon Extended Cab 2012', 'Geo Metro Convertible 1993', 'HUMMER H3T Crew Cab 2010', 'HUMMER H2 SUT Crew Cab 2009', 'Honda Odyssey Minivan 2012', 'Honda Odyssey Minivan 2007', 'Honda Accord Coupe 2012', 'Honda Accord Sedan 2012', 'Hyundai Veloster Hatchback 2012', 'Hyundai Santa Fe SUV 2012', 'Hyundai Tucson SUV 2012', 'Hyundai Veracruz SUV 2012', 'Hyundai Sonata Hybrid Sedan 2012', 'Hyundai Elantra Sedan 2007', 'Hyundai Accent Sedan 2012', 'Hyundai Genesis Sedan 2012', 'Hyundai Sonata Sedan 2012', 'Hyundai Elantra Touring Hatchback 2012', 'Hyundai Azera Sedan 2012', 'Infiniti G Coupe IPL 2012', 'Infiniti QX56 SUV 2011', 'Isuzu Ascender SUV 2008', 'Jaguar XK XKR 2012', 'Jeep Patriot SUV 2012', 'Jeep Wrangler SUV 2012', 'Jeep Liberty SUV 2012', 'Jeep Grand Cherokee SUV 2012', 'Jeep Compass SUV 2012', 'Lamborghini Reventon Coupe 2008', 'Lamborghini Aventador Coupe 2012', 'Lamborghini Gallardo LP 570-4 Superleggera 2012', 'Lamborghini Diablo Coupe 2001', 'Land Rover Range Rover SUV 2012', 'Land Rover LR2 SUV 2012', 'Lincoln Town Car Sedan 2011', 'MINI Cooper Roadster Convertible 2012', 'Maybach Landaulet Convertible 2012', 'Mazda Tribute SUV 2011', 'McLaren MP4-12C Coupe 2012', 'Mercedes-Benz 300-Class Convertible 1993', 'Mercedes-Benz C-Class Sedan 2012', 'Mercedes-Benz SL-Class Coupe 2009', 'Mercedes-Benz E-Class Sedan 2012', 'Mercedes-Benz S-Class Sedan 2012', 'Mercedes-Benz Sprinter Van 2012', 'Mitsubishi Lancer Sedan 2012', 'Nissan Leaf Hatchback 2012', 'Nissan NV Passenger Van 2012', 'Nissan Juke Hatchback 2012', 'Nissan 240SX Coupe 1998', 'Plymouth Neon Coupe 1999', 'Porsche Panamera Sedan 2012', 'Ram CV Cargo Van Minivan 2012', 'Rolls-Royce Phantom Drophead Coupe Convertible 2012', 'Rolls-Royce Ghost Sedan 2012', 'Rolls-Royce Phantom Sedan 2012', 'Scion xD Hatchback 2012', 'Spyker C8 Convertible 2009', 'Spyker C8 Coupe 2009', 'Suzuki Aerio Sedan 2007', 'Suzuki Kizashi Sedan 2012', 'Suzuki SX4 Hatchback 2012', 'Suzuki SX4 Sedan 2012', 'Tesla Model S Sedan 2012', 'Toyota Sequoia SUV 2012', 'Toyota Camry Sedan 2012', 'Toyota Corolla Sedan 2012', 'Toyota 4Runner SUV 2012', 'Volkswagen Golf Hatchback 2012', 'Volkswagen Golf Hatchback 1991', 'Volkswagen Beetle Hatchback 2012', 'Volvo C30 Hatchback 2012', 'Volvo 240 Sedan 1993', 'Volvo XC90 SUV 2007', 'smart fortwo Convertible 2012']
        
        id = name_list.index(label)
        return id


def fetch_dataloader(types, params):
#     """
#     Fetches the DataLoader object for each type in types from data_dir.

#     Args:
#         types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
#         data_dir: (string) directory containing the dataset
#         params: (Params) hyperparameters

#     Returns:
#         data: (dict) contains the DataLoader object for each type in types
#     """
    dataloaders = {}
    name_list = ['AM General Hummer SUV 2000', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012',
         'Acura TL Type-S 2008', 'Acura TSX Sedan 2012', 'Acura Integra Type R 2001',
         'Acura ZDX Hatchback 2012', 'Aston Martin V8 Vantage Convertible 2012',
         'Aston Martin V8 Vantage Coupe 2012', 'Aston Martin Virage Convertible 2012',
         'Aston Martin Virage Coupe 2012', 'Audi RS 4 Convertible 2008', 'Audi A5 Coupe 2012',
         'Audi TTS Coupe 2012', 'Audi R8 Coupe 2012', 'Audi V8 Sedan 1994', 'Audi 100 Sedan 1994',
         'Audi 100 Wagon 1994', 'Audi TT Hatchback 2011', 'Audi S6 Sedan 2011', 'Audi S5 Convertible 2012',
         'Audi S5 Coupe 2012', 'Audi S4 Sedan 2012', 'Audi S4 Sedan 2007', 'Audi TT RS Coupe 2012',
         'BMW ActiveHybrid 5 Sedan 2012', 'BMW 1 Series Convertible 2012', 'BMW 1 Series Coupe 2012',
         'BMW 3 Series Sedan 2012', 'BMW 3 Series Wagon 2012', 'BMW 6 Series Convertible 2007',
         'BMW X5 SUV 2007', 'BMW X6 SUV 2012', 'BMW M3 Coupe 2012', 'BMW M5 Sedan 2010',
         'BMW M6 Convertible 2010', 'BMW X3 SUV 2012', 'BMW Z4 Convertible 2012',
         'Bentley Continental Supersports Conv. Convertible 2012', 'Bentley Arnage Sedan 2009',
         'Bentley Mulsanne Sedan 2011', 'Bentley Continental GT Coupe 2012',
         'Bentley Continental GT Coupe 2007', 'Bentley Continental Flying Spur Sedan 2007',
         'Bugatti Veyron 16.4 Convertible 2009', 'Bugatti Veyron 16.4 Coupe 2009',
         'Buick Regal GS 2012', 'Buick Rainier SUV 2007', 'Buick Verano Sedan 2012',
         'Buick Enclave SUV 2012', 'Cadillac CTS-V Sedan 2012', 'Cadillac SRX SUV 2012',
         'Cadillac Escalade EXT Crew Cab 2007', 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',
         'Chevrolet Corvette Convertible 2012', 'Chevrolet Corvette ZR1 2012',
         'Chevrolet Corvette Ron Fellows Edition Z06 2007', 'Chevrolet Traverse SUV 2012',
         'Chevrolet Camaro Convertible 2012', 'Chevrolet HHR SS 2010', 'Chevrolet Impala Sedan 2007',
         'Chevrolet Tahoe Hybrid SUV 2012', 'Chevrolet Sonic Sedan 2012',
         'Chevrolet Express Cargo Van 2007', 'Chevrolet Avalanche Crew Cab 2012',
         'Chevrolet Cobalt SS 2010', 'Chevrolet Malibu Hybrid Sedan 2010',
         'Chevrolet TrailBlazer SS 2009', 'Chevrolet Silverado 2500HD Regular Cab 2012', 'Chevrolet Silverado 1500 Classic Extended Cab 2007', 'Chevrolet Express Van 2007', 'Chevrolet Monte Carlo Coupe 2007', 'Chevrolet Malibu Sedan 2007', 'Chevrolet Silverado 1500 Extended Cab 2012', 'Chevrolet Silverado 1500 Regular Cab 2012', 'Chrysler Aspen SUV 2009', 'Chrysler Sebring Convertible 2010', 'Chrysler Town and Country Minivan 2012', 'Chrysler 300 SRT-8 2010', 'Chrysler Crossfire Convertible 2008', 'Chrysler PT Cruiser Convertible 2008', 'Daewoo Nubira Wagon 2002', 'Dodge Caliber Wagon 2012', 'Dodge Caliber Wagon 2007', 'Dodge Caravan Minivan 1997', 'Dodge Ram Pickup 3500 Crew Cab 2010', 'Dodge Ram Pickup 3500 Quad Cab 2009', 'Dodge Sprinter Cargo Van 2009', 'Dodge Journey SUV 2012', 'Dodge Dakota Crew Cab 2010', 'Dodge Dakota Club Cab 2007', 'Dodge Magnum Wagon 2008', 'Dodge Challenger SRT8 2011', 'Dodge Durango SUV 2012', 'Dodge Durango SUV 2007', 'Dodge Charger Sedan 2012', 'Dodge Charger SRT-8 2009', 'Eagle Talon Hatchback 1998', 'FIAT 500 Abarth 2012', 'FIAT 500 Convertible 2012', 'Ferrari FF Coupe 2012', 'Ferrari California Convertible 2012', 'Ferrari 458 Italia Convertible 2012', 'Ferrari 458 Italia Coupe 2012', 'Fisker Karma Sedan 2012', 'Ford F-450 Super Duty Crew Cab 2012', 'Ford Mustang Convertible 2007', 'Ford Freestar Minivan 2007', 'Ford Expedition EL SUV 2009', 'Ford Edge SUV 2012', 'Ford Ranger SuperCab 2011', 'Ford GT Coupe 2006', 'Ford F-150 Regular Cab 2012', 'Ford F-150 Regular Cab 2007', 'Ford Focus Sedan 2007', 'Ford E-Series Wagon Van 2012', 'Ford Fiesta Sedan 2012', 'GMC Terrain SUV 2012', 'GMC Savana Van 2012', 'GMC Yukon Hybrid SUV 2012', 'GMC Acadia SUV 2012', 'GMC Canyon Extended Cab 2012', 'Geo Metro Convertible 1993', 'HUMMER H3T Crew Cab 2010', 'HUMMER H2 SUT Crew Cab 2009', 'Honda Odyssey Minivan 2012', 'Honda Odyssey Minivan 2007', 'Honda Accord Coupe 2012', 'Honda Accord Sedan 2012', 'Hyundai Veloster Hatchback 2012', 'Hyundai Santa Fe SUV 2012', 'Hyundai Tucson SUV 2012', 'Hyundai Veracruz SUV 2012', 'Hyundai Sonata Hybrid Sedan 2012', 'Hyundai Elantra Sedan 2007', 'Hyundai Accent Sedan 2012', 'Hyundai Genesis Sedan 2012', 'Hyundai Sonata Sedan 2012', 'Hyundai Elantra Touring Hatchback 2012', 'Hyundai Azera Sedan 2012', 'Infiniti G Coupe IPL 2012', 'Infiniti QX56 SUV 2011', 'Isuzu Ascender SUV 2008', 'Jaguar XK XKR 2012', 'Jeep Patriot SUV 2012', 'Jeep Wrangler SUV 2012', 'Jeep Liberty SUV 2012', 'Jeep Grand Cherokee SUV 2012', 'Jeep Compass SUV 2012', 'Lamborghini Reventon Coupe 2008', 'Lamborghini Aventador Coupe 2012', 'Lamborghini Gallardo LP 570-4 Superleggera 2012', 'Lamborghini Diablo Coupe 2001', 'Land Rover Range Rover SUV 2012', 'Land Rover LR2 SUV 2012', 'Lincoln Town Car Sedan 2011', 'MINI Cooper Roadster Convertible 2012', 'Maybach Landaulet Convertible 2012', 'Mazda Tribute SUV 2011', 'McLaren MP4-12C Coupe 2012', 'Mercedes-Benz 300-Class Convertible 1993', 'Mercedes-Benz C-Class Sedan 2012', 'Mercedes-Benz SL-Class Coupe 2009', 'Mercedes-Benz E-Class Sedan 2012', 'Mercedes-Benz S-Class Sedan 2012', 'Mercedes-Benz Sprinter Van 2012', 'Mitsubishi Lancer Sedan 2012', 'Nissan Leaf Hatchback 2012', 'Nissan NV Passenger Van 2012', 'Nissan Juke Hatchback 2012', 'Nissan 240SX Coupe 1998', 'Plymouth Neon Coupe 1999', 'Porsche Panamera Sedan 2012', 'Ram CV Cargo Van Minivan 2012', 'Rolls-Royce Phantom Drophead Coupe Convertible 2012', 'Rolls-Royce Ghost Sedan 2012', 'Rolls-Royce Phantom Sedan 2012', 'Scion xD Hatchback 2012', 'Spyker C8 Convertible 2009', 'Spyker C8 Coupe 2009', 'Suzuki Aerio Sedan 2007', 'Suzuki Kizashi Sedan 2012', 'Suzuki SX4 Hatchback 2012', 'Suzuki SX4 Sedan 2012', 'Tesla Model S Sedan 2012', 'Toyota Sequoia SUV 2012', 'Toyota Camry Sedan 2012', 'Toyota Corolla Sedan 2012', 'Toyota 4Runner SUV 2012', 'Volkswagen Golf Hatchback 2012', 'Volkswagen Golf Hatchback 1991', 'Volkswagen Beetle Hatchback 2012', 'Volvo C30 Hatchback 2012', 'Volvo 240 Sedan 1993', 'Volvo XC90 SUV 2007', 'smart fortwo Convertible 2012']
        

    for split in ['train', 'val']:
        if split in types:
            train_path = os.path.join("/Users/roshanbisht/Documents/code/assignments/godigit/data_splits/train")
            val_path = os.path.join("/Users/roshanbisht/Documents/code/assignments/godigit/data_splits/val")

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                # dl = DataLoader(OCRClassifierDataset(train_path, train_transformer), batch_size=params.batch_size, shuffle=True,
                #                         num_workers=params.num_workers,
                #                         pin_memory=params.cuda)
                train_dataset = CarClassifierDataset(train_path, train_transformer)
                class_weights = [0] * 196

                for folder_name in os.listdir(train_path):
                    try:
                        filenames = [os.path.join(train_path, folder_name, f) for f in os.listdir(os.path.join(train_path, folder_name)) if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
                    except:
                        print("folder_name: ", folder_name)
                        continue
                    # print("folder_name: ", folder_name)
                    id = name_list.index(folder_name)
                    class_weights[id] = 1/len(filenames)

                sample_weights = [0] * len(train_dataset)
                for idx, (data, label) in enumerate(train_dataset):
                    class_weight = class_weights[label]
                    sample_weights[idx] = class_weight

                sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),\
                                                replacement=True)
                dl = DataLoader(train_dataset, batch_size=params.batch_size, sampler=sampler, 
                                num_workers=params.num_workers, pin_memory=params)

            else:
                val_dataset =  CarClassifierDataset(val_path, eval_transformer)
                class_weights = [0] * 196

                for folder_name in os.listdir(val_path):
                    try:
                        filenames = [os.path.join(val_path, folder_name, f) for f in os.listdir(os.path.join(val_path, folder_name)) if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
                    except:
                        print("folder_name: ", folder_name)
                        continue
                    id = name_list.index(folder_name)
                    class_weights[id] = 1/len(filenames)

                sample_weights = [0] * len(val_dataset)
                for idx, (data, label) in enumerate(val_dataset):
                    class_weight = class_weights[label]
                    sample_weights[idx] = class_weight
                    
                sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),\
                                                replacement=True)
                dl = DataLoader(val_dataset, batch_size=params.batch_size,sampler=sampler
                                ,num_workers=params.num_workers, pin_memory=params)

            dataloaders[split] = dl

    return dataloaders


if __name__=='__main__':
    # print("hello")
    data_dir = "/home/data/ocr_data_mini"
    # foldernames = os.listdir(data_dir)

    # filenames = [os.path.join(data_dir, f) for f in foldernames if f.endswith('.jpg')]
    # [os.path.join(data_dir, folder_name, f) for f in os.listdir(os.path.join(data_dir)) if f.endswith('.jpeg')]]
    # print("self.filenames:", self.filenames)
    # all_file_names = []
    # for folder_name in os.listdir(data_dir):
    #     # print(folder_name)
    #     filenames = [os.path.join(data_dir, folder_name, f) for f in os.listdir(os.path.join(data_dir, folder_name)) if f.endswith('.jpeg') or f.endswith('.jpg')]
    #     all_file_names += filenames
    # labels = [os.path.split(os.path.split(filename)[0])[-1] for filename in all_file_names]
    # # self.transform = transform
    # print(len(all_file_names))
    # # labels = os.path.split(os.path.split(all_file_names[0])[0])[-1]
    # print(len(labels))

    # dataset = OCRClassifierDataset(data_dir, )

