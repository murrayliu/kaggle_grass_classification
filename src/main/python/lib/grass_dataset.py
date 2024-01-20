from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import sys
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms


class GrassDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.classes = sorted(os.listdir(data_path))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}
        self.images = self.load_images()

    def load_images(self):
        images = []
        for cls in self.classes:
            class_path = os.path.join(self.data_path, cls)
            class_idx = self.class_to_idx[cls]
            for filename in os.listdir(class_path):
                image_path = os.path.join(class_path, filename)
                images.append((image_path, class_idx))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        # === transform ===
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image = transform(image)

        return image, label
    

class GrassTrainTestDataloader(object):
    def __init__(self, data_path, split_ratio, batch_size, shuffle=True):
        self.data_path = data_path
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataloader(self):
        self.full_dataset = GrassDataset(data_path=self.data_path)
        
        # === Split the dataset ===
        train_size = int(self.split_ratio * len(self.full_dataset)) 
        test_size = len(self.full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.full_dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return train_loader, test_loader
    
    def get_idx_to_classstr_dict(self):

        return self.full_dataset.idx_to_class

    