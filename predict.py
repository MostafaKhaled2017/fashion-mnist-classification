#Importing main libraries
import os
import torch
from  lightningmodule import FashionMNISTDataClassifier
from datamodule import FashionMNISTDataModule

import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config", version_base= None)
def main(cfg: DictConfig):
    model = FashionMNISTDataClassifier(cfg)
    dm = FashionMNISTDataModule()
    
    # Load the state dict
    model.load_state_dict(torch.load("model.ckpt"))

    img, label = dm.get_test_image()

    label = torch.tensor(label) if not isinstance(label, torch.Tensor) else label

    # Calculate the class probabilities (softmax) for img
    proba = torch.exp(model(img))

    # Plot the image and probabilities
    desc = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    fig, (ax1, ax2) =  plt.subplots(figsize=(13, 6), nrows=1, ncols=2)
    ax1.axis('off')
    ax1.imshow(img.squeeze().reshape(28, 28))
    ax1.set_title(desc[label.item()])
    ax2.bar(range(10), proba.detach().cpu().numpy().squeeze())
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(desc, size='small')
    ax2.set_title('Predicted Probabilities')
    plt.tight_layout()
    plt.show()


main()