#Importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class FashionMNISTDataClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.network = nn.Sequential(OrderedDict([('fc1', nn.Linear(784, 392)),
                                       ('relu1', nn.ReLU()),
                                       ('drop1', nn.Dropout(0.25)),
                                       ('fc12', nn.Linear(392, 196)),
                                       ('relu2', nn.ReLU()),
                                       ('drop2', nn.Dropout(0.25)),
                                       ('fc3', nn.Linear(196, 98)),
                                       ('relu3', nn.ReLU()),
                                       ('drop3', nn.Dropout(0.25)),                                       
                                       ('fc4', nn.Linear(98, 49)),
                                       ('relu4', nn.ReLU()),
                                       ('output', nn.Linear(49, 10)),
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
        optimizer = None
        if self.cfg.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr)
        elif optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr = self.cfg.lr)

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