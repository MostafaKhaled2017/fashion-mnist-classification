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

#Data Module Class
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
    

#Lightning Model Class
class FashionMNISTDataClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(OrderedDict([('fc1', nn.Linear(784, 128)),
                                                ('relu1', nn.ReLU()),
                                                ('drop1', nn.Dropout(0.25)),
                                                ('fc2', nn.Linear(128, 64)),
                                                ('relu2', nn.ReLU()),
                                                ('drop1', nn.Dropout(0.25)),
                                                ('output', nn.Linear(64, 10)),
                                                ('logsoftmax', nn.LogSoftmax(dim=1))]))
        self.loss_fn = nn.NLLLoss()

        self.val_accuracy = Accuracy(task='multiclass', num_classes=10)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=10)

        #Will be used for storing the average losses after each epoch
        self.train_losses =[]
        self.test_losses = []

        #Will be used for storing the losses after each step
        self.temp_training_losses = []
        self.temp_validation_losses = []

        
    def forward(self, x):
        x = self.network(x)
        return x
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.view(images.shape[0], -1)
        out = self.forward(images)
        loss = self.loss_fn(out, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.temp_training_losses.append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.view(images.shape[0], -1)
        out = self.forward(images)
        loss = self.loss_fn(out, labels)
        preds = torch.argmax(out, dim=1)
        self.val_accuracy.update(preds, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)

        self.temp_validation_losses.append(loss)
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        images = images.view(images.shape[0], -1)
        out = self.forward(images)
        loss = self.loss_fn(out, labels)
        preds = torch.argmax(out, dim=1)
        self.test_accuracy.update(preds, labels)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.003)
        return optimizer
    
    def on_train_epoch_end(self, *args, **kwargs):        
        train_loss_mean = torch.stack(self.temp_training_losses).mean()
        self.temp_training_losses = []
        self.train_losses.append(train_loss_mean)


    def on_validation_epoch_end(self, *arg, **kwargs):
        val_loss_mean = torch.stack(self.temp_validation_losses).mean()
        self.temp_validation_losses = []
        self.test_losses.append(val_loss_mean)

    def plot_losses(self):
        train_losses_tensor = torch.tensor(self.train_losses, dtype=torch.float32)
        test_losses_tensor = torch.tensor(self.test_losses, dtype=torch.float32)
        
        plt.plot(train_losses_tensor.detach().numpy(), label='train-loss')
        plt.plot(test_losses_tensor.detach().numpy(), label='val-loss')
        plt.legend()
        plt.show()


#Initializing the data module and the model
dm = FashionMNISTDataModule()
model = FashionMNISTDataClassifier()

