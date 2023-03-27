#Importing libraries
from torchvision import transforms, datasets
import torch
import numpy as np
import pytorch_lightning as pl

class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
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
    
    #Used in predict.py
    def get_test_image(self):
        # define transforms
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
        test_dataset = datasets.FashionMNIST(self.data_dir, download=True, train=False, transform=transform)
        index = 0
        img, label = test_dataset[index]
        img = img.view(img.shape[0], -1)
        return img, label
    
    def val_dataloader(self):
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.val_idx)
        val_dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=64, sampler=val_sampler)
        return val_dl
    
    def test_dataloader(self):
        test_dl = torch.utils.data.DataLoader(self.test_dataset, batch_size=64, shuffle=True)
        return test_dl