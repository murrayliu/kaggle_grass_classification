from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import sys
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import cv2
from matplotlib import pyplot as plt

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
        image_path, label = self.images[idx]
        # image = Image.open(img_path).convert("RGB")
        image = self.preprocess_image(image_path)
        
        # === transform ===
        transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(degrees=30),
        ]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomApply(transform_list, p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image)

        return image, label
    
    def preprocess_image(self, image_path):
        bgr_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        blur_image = cv2.GaussianBlur(bgr_image, (5, 5), 0)
        hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

        # === extract green area ===
        lower = (25,40,50)
        upper = (75,255,255)
        mask_image = cv2.inRange(hsv_image,lower,upper)
        struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        blur_mask_image = cv2.morphologyEx(mask_image,cv2.MORPH_CLOSE, struc)

        # === mask raw image ===
        boolean = blur_mask_image == 0
        rgb_image[boolean] = (0 ,0, 0)
        rgb_image = Image.fromarray(rgb_image)

        # plt.imshow(rgb_image)
        # plt.show()
        return rgb_image

class GrassTrainTestDataloader(object):
    def __init__(self, data_path, split_ratio, batch_size, shuffle=True):
        self.data_path = data_path
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.full_dataset = GrassDataset(data_path=self.data_path)

    def get_dataloader(self):
        # === Split the dataset ===
        train_size = int(self.split_ratio * len(self.full_dataset)) 
        test_size = len(self.full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.full_dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return train_loader, test_loader
    
    def get_idx_to_classstr_dict(self):

        return self.full_dataset.idx_to_class

    