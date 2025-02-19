#Importing main libraries
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

import hydra
from omegaconf import DictConfig

from  datamodule import FashionMNISTDataModule
from  lightningmodule import FashionMNISTDataClassifier    


@hydra.main(config_path="config", config_name="config", version_base= None)
def main(cfg: DictConfig):
    #Initializing the data module and the model
    dm = FashionMNISTDataModule()
    model = FashionMNISTDataClassifier(cfg)

    #Training the Network
    trainer = pl.Trainer(max_epochs=cfg.epochs, callbacks=[TQDMProgressBar(refresh_rate=20)],
        logger=CSVLogger(save_dir="logs/"))

    trainer.fit(model, dm)

    # Validate
    trainer.test(model, datamodule=dm)

    # Save the state dict
    torch.save(model.state_dict(), "model.ckpt")


main()