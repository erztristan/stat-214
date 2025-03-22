import numpy as np
import sys
import os
import yaml  # pip install pyyaml
import gc
import torch
import lightning as L

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from autoencoder import Autoencoder
from patchdataset import PatchDataset
from pytorch_lightning.callbacks import EarlyStopping

def run_autoencoder(data_path, config_path, patches):
    print("loading config file")
    #config_path = sys.argv[2]
    assert os.path.exists(config_path), f"Config file {config_path} not found"
    config = yaml.safe_load(open(config_path, "r"))

    # clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    print("making the patch data")

    # get the patches
    #_, patches, _ = make_data(data_path, patch_size=config["data"]["patch_size"])

    """
    In the original code, all images were patched and integrated into a single list, 
    and then randomly divided into test data and training data, so the same images 
    were used in both the test data and the training data. 
    This meant that data was leaked, and it could reduce the generalization performance. 
    Therefore, we prevented data leakage by patching each image and then selecting 
    the images to be used for the test data and training data respectively.
    """
    
    np.random.seed(42)
    train_bool = np.random.rand(len(patches)) < 0.8
    train_idx = np.where(train_bool)[0]
    val_idx = np.where(~train_bool)[0]

    print("train:", train_idx)
    print("val: ", val_idx)

    train_container = [patches[i] for i in train_idx]
    val_container = [patches[i] for i in val_idx]
    train_patches = [patch for image_patches in train_container for patch in image_patches]
    val_patches = [patch for image_patches in val_container for patch in image_patches]
    train_dataset = PatchDataset(train_patches)
    val_dataset = PatchDataset(val_patches)

    # create train and val dataloaders
    dataloader_train = DataLoader(train_dataset, **config["dataloader_train"])
    dataloader_val = DataLoader(val_dataset, **config["dataloader_val"])

    print("initializing model")
    # Initialize an autoencoder object
    model = Autoencoder(
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )
    print(model)

    print("preparing for training")
    # configure the settings for making checkpoints
    checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

    # if running in slurm, add slurm job id info to the config file
    if "SLURM_JOB_ID" in os.environ:
        config["slurm_job_id"] = os.environ["SLURM_JOB_ID"]

    # initialize the wandb logger, giving it our config file
    # to save, and also configuring the logger itself.
    wandb_logger = WandbLogger(config=config, **config["wandb"])

    # initialize the trainer
    trainer = L.Trainer(
        logger=wandb_logger, callbacks=[checkpoint_callback], **config["trainer"]
    )

    print("training")
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # clean up memory
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    print("ready")
    run_autoencoder()

