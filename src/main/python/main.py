import os
import sys
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from lib.grass_dataset import GrassDataset, GrassTrainTestDataloader
from lib.grass_classification import GrassClassification, CfgEnum, read_cfg
from lib.model import GrassCNN


class ExecutionKernel(object):
    def __init__(self, cfg_file, dataloader_, model):
        self.cfg_file = cfg_file
        self.dataloader_ = dataloader_
        self.model = model

    def train(self):
        self.grass_class_algo = GrassClassification(cfg_file=self.cfg_file, dataloader_=self.dataloader_, model=self.model)
        self.grass_class_algo.process() 
    
    def eval(self, test_data_path):
        self.grass_class_algo.prediction(test_data_path=test_data_path)
    
if __name__ == "__main__":

    # === read parameter ===
    cfg_path = "C:/Users/murray/Desktop/kaggle_grass_classification/src/main/python/config/grass_classification.yaml"
    grass_cfg = read_cfg(cfg_path=cfg_path)
    
    # === read data ===
    train_data_path = 'C:/Users/murray/Desktop/kaggle_grass_classification/src/main/python/data/train'
    test_data_path = 'C:/Users/murray/Desktop/kaggle_grass_classification/src/main/python/data/test'

    grass_loader = GrassTrainTestDataloader(data_path=train_data_path, 
                                            split_ratio=grass_cfg[CfgEnum.split_ratio],
                                            batch_size=grass_cfg[CfgEnum.batch_size]
                                           )
    # === read model ===
    grass_model = GrassCNN(num_classes=grass_cfg[CfgEnum.num_classes])

    # === start training ===
    grass_kernel = ExecutionKernel(cfg_file=grass_cfg, dataloader_=grass_loader, model=grass_model)
    grass_kernel.train()
    
    # === start evaluation ===
    grass_kernel.eval(test_data_path=test_data_path)