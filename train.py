#Importing main libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import OrderedDict
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torchmetrics import Accuracy
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

#Data Module class
class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'F_MNIST_data'):
        super().__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        # download the dataset if it does not exist
        datasets.FashionMNIST(self.data_dir, train=True, download=True)
        datasets.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # define transforms
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
        
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.FashionMNIST(self.data_dir, download=True, train=True, transform=transform)
            train_num = len(self.train_dataset)
            indices = list(range(train_num))
            np.random.shuffle(indices)
            split = int(np.floor(0.2 * train_num))
            self.val_idx, self.train_idx = indices[:split], indices[split:]
            

        if stage == 'test' or stage is None:
            self.test_dataset = datasets.FashionMNIST(self.data_dir, download=True, train=False, transform=transform)

    def train_dataloader(self):
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_idx)
        train_dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=64, sampler=train_sampler)
        return train_dl
    
    def val_dataloader(self):
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.val_idx)
        val_dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=64, sampler=val_sampler)
        return val_dl
    
    def test_dataloader(self):
        test_dl = torch.utils.data.DataLoader(self.test_dataset, batch_size=64, shuffle=True)
        return test_dl
    

# Lightning Module