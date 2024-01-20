import os
import glob
import yaml
import torch
import numpy as np
import torch.nn as nn
from enum import Enum
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import f1_score

class GrassClassification(object):
    def __init__(self, cfg_file, dataloader_, model):
        
        # === training setting ===
        self.cfg = cfg_file
        self.dataloader_ = dataloader_
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg[CfgEnum.learning_rate])
        
        self.model_setting_str = "bs{}_lr{}_nepochs{}".format(self.cfg[CfgEnum.batch_size], 
                                                             self.cfg[CfgEnum.learning_rate], 
                                                             self.cfg[CfgEnum.num_epochs]
                                                            )
        
        self.review_path = "./output/review/{}/{}".format(datetime.today().strftime('%Y-%m-%d'),
                                                          self.model_setting_str)
        self.weight_path = '{}/{}.pth'.format(self.review_path, self.model_setting_str)

        print('Device: {}'.format(self.device))

    def process(self):
        # === read loader ===
        train_loader, test_loader = self.dataloader_.get_dataloader()

        # === start training ===
        best_epoch = 0
        best_f1 = 0
        early_stop_counts = self.cfg[CfgEnum.early_stop_counts]
        train_f1_list = []
        test_f1_list = []
        for cur_epoch in range(self.cfg[CfgEnum.num_epochs]):
   
            train_f1 = self.train_session(train_loader=train_loader, cur_epoch=cur_epoch)
            test_f1 = self.test_session(test_loader=test_loader)
            train_f1_list.append(train_f1)
            test_f1_list.append(test_f1)
            self.review_figure(cur_epoch, best_epoch, train_f1_list, test_f1_list)

            # === save best model ===
            if test_f1 > best_f1:
                early_stop_counts = self.cfg[CfgEnum.early_stop_counts]
                best_epoch = cur_epoch
                best_f1 = test_f1
                torch.save(self.model.state_dict(), self.weight_path)

            # === ealry stop ===
            early_stop_counts -= 1
            if early_stop_counts == 0:
                break

        return train_f1_list, test_f1_list
    
    def train_session(self, train_loader, cur_epoch):
        self.model.train()
        all_preds = []
        all_labels = []
        total_batches = len(train_loader)
        pbar = tqdm(total=total_batches, desc=f"Epoch {cur_epoch + 1}/{self.cfg[CfgEnum.num_epochs]}", unit="batch")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.update(1)  
        pbar.set_postfix({"Loss": loss.item()})
        pbar.close()  

        f1_score_ = f1_score(all_labels, all_preds, average='macro')
        print(f"f1_score on the train set: {f1_score_ * 100:.2f}%")

        return f1_score_

    def test_session(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            total_batches = len(test_loader)
            pbar = tqdm(total=total_batches, desc="Testing", unit="batch")
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.update(1) 
            pbar.close()  

        f1_score_ = f1_score(all_labels, all_preds, average='macro')
        print(f"F1-score on the test set: {f1_score_ * 100:.2f}%")

        return f1_score_

    def prediction(self, test_data_path):

        # === load test image ===
        file_pattern = '**/*.png'
        full_pattern = '{}/{}'.format(test_data_path, file_pattern)
        image_filenames = glob.glob(full_pattern, recursive=True)

        # === load best model ===
        self.model.load_state_dict(torch.load(self.weight_path))
        self.model.eval() 

        # === start evulation ===
        submission_list = []
        for image_filename in image_filenames:

            # --- read image basename ---
            base_filename = os.path.basename(image_filename)

            # --- read image tensor ---
            raw_image = Image.open(image_filename).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            raw_image = transform(raw_image)

            # --- loop over all test images ---
            idx_to_classstr_dict = self.dataloader_.get_idx_to_classstr_dict()

            with torch.no_grad():
                raw_image = torch.unsqueeze(raw_image, 0) # set batch_size=1 to fit model
                raw_image = raw_image.to(self.device)
                outputs = self.model(raw_image)
                _, predicted = torch.max(outputs, 1)
                predict_numpy = predicted.cpu().numpy()
                predict_list = [base_filename, idx_to_classstr_dict[int(predict_numpy[0])]]
            
            submission_list.append(predict_list)

        # === output to csv ===
        csv_file_path = 'submission.csv'
        csv_review_path = '{}/{}'.format(self.review_path, csv_file_path)
        headers = ['file', 'species']
        np.savetxt(csv_review_path, submission_list, delimiter=',', fmt='%s', header=','.join(headers), comments='')
        print(f'Data has been written to {csv_review_path}')

    def review_figure(self, cur_epoch, best_epoch, train_f1_list, test_f1_list):

        # === make folder ===
        figure_name = '{}.png'.format(self.model_setting_str)
        figure_path = os.path.join(self.review_path, figure_name)
        if not os.path.exists(self.review_path):
            os.makedirs(self.review_path)

        # === save f1_score figure ===
        epoch_idx = range(cur_epoch + 1)
        plt.figure() 
        plt.plot(epoch_idx, train_f1_list, label="train f1-score",  color='blue')
        plt.plot(epoch_idx, test_f1_list, label="test f1-score",  color='red')
        plt.title('Grass classification (best epoch: {})'.format(best_epoch))
        plt.xlabel('Epoch')
        plt.ylabel('f1-score')
        plt.xlim(0, self.cfg[CfgEnum.num_epochs])
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(figure_path)
        plt.close()

    


class CfgEnum(str, Enum):
    # === dataloader ===
    split_ratio = "split_ratio"
    batch_size = "batch_size"
    num_classes = "num_classes"
    learning_rate = "learning_rate"
    num_epochs = "num_epochs"
    early_stop_counts = "early_stop_counts"

def read_cfg(cfg_path):
    with open(cfg_path, 'r') as file:
        data = yaml.safe_load(file)
    data = data["grass"]
    return data