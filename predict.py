#Importing main libraries
import os
import torch
from  lightningmodule import FashionMNISTDataClassifier
from datamodule import FashionMNISTDataModule

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config", version_base= None)
def main(cfg: DictConfig):
    model = FashionMNISTDataClassifier(cfg)
    dm = FashionMNISTDataModule()
    
    # Load the state dict
    model.load_state_dict(torch.load("model.ckpt"))

    img, label = dm.get_test_image()

    # Calculate the class probabilities (softmax) for img
    proba = torch.exp(model(img))

    print(proba)



main()