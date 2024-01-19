import yaml
import torch
import torch.nn as nn
from enum import Enum
from tqdm import tqdm
import torch.optim as optim

class GrassClassification(object):
    def __init__(self, cfg_file, dataloader_, model):
        self.cfg = cfg_file
        self.dataloader_ = dataloader_
        self.model = model
        
        # === training setting ===
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg[CfgEnum.learning_rate])

        print(self.device)
    def process(self):
        # === read loader ===
        train_loader, test_loader = self.dataloader_.get_dataloader()

        # === start training ===
        for cur_epoch in range(self.cfg[CfgEnum.num_epochs]):
            # Set the model to training mode
            self.train_session(train_loader=train_loader, cur_epoch=cur_epoch)
            res = self.test_session(test_loader=test_loader)
        
        return res
    
    def train_session(self, train_loader, cur_epoch):
        self.model.train()
        total_batches = len(train_loader)
        pbar = tqdm(total=total_batches, desc=f"Epoch {cur_epoch + 1}/{self.cfg[CfgEnum.num_epochs]}", unit="batch")
        for inputs, labels in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            pbar.update(1)  
        pbar.set_postfix({"Loss": loss.item()})
        pbar.close()  

    def test_session(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            total_batches = len(test_loader)
            pbar = tqdm(total=total_batches, desc="Testing", unit="batch")
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.update(1) 
            pbar.close()  

        accuracy = correct / total
        print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

        return accuracy
    
class CfgEnum(str, Enum):
    # === dataloader ===
    split_ratio = "split_ratio"
    batch_size = "batch_size"
    num_classes = "num_classes"
    learning_rate = "learning_rate"
    num_epochs = "num_epochs"

def read_cfg(cfg_path):
    with open(cfg_path, 'r') as file:
        data = yaml.safe_load(file)
    data = data["grass"]
    return data