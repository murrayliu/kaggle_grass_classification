import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# === Define the CNN model ===
import torch.nn as nn

class GrassCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(GrassCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.1)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.relu5 = nn.ReLU()
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.relu6 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 28 * 28, 256)
        self.relu7 = nn.ReLU()
        self.batchnorm7 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 256)
        self.relu8 = nn.ReLU()
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.dropout1(self.batchnorm2(self.maxpool1(self.relu2(self.conv2(self.batchnorm1(self.relu1(self.conv1(x))))))))
        x = self.dropout2(self.batchnorm4(self.maxpool2(self.relu4(self.conv4(self.batchnorm3(self.relu3(self.conv3(x))))))))
        x = self.dropout3(self.batchnorm6(self.maxpool3(self.relu6(self.conv6(self.batchnorm5(self.relu5(self.conv5(x))))))))
        x = self.flatten(x)
        x = self.dropout4(self.batchnorm7(self.relu7(self.fc1(x))))
        x = self.dropout5(self.batchnorm8(self.relu8(self.fc2(x))))
        x = self.fc3(x)
        return x
