import argparse
from typing import Tuple

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.lit_models import LitModel


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )
    return train_transform, test_transform


def make_callback() -> list:
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename="{epoch}-{val_acc:.2f}")
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )
    return [early_stop_callback, checkpoint_callback, progress_bar, lr_monitor]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--lr", default=0.05, type=float, help="learning rate")
    parser.add_argument(
        "--resume", "-r", type=str, default=None, help="resume from checkpoint"
    )
    args = parser.parse_args()
    wandb.login()

    seed_everything(42)

    train_transform, test_transform = make_transforms()
    data_module = CIFAR10DataModule(data_dir="./data", batch_size=64, num_workers=4)
    data_module.train_transforms = train_transform
    data_module.test_transforms = test_transform
    data_module.val_transforms = test_transform

    data_module.prepare_data()
    data_module.setup()

    model = LitModel(lr=args.lr, batch_size=64)

    if args.resume:
        model.load_from_checkpoint(args.resume)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=WandbLogger(project="wandb-lightning", job_type="train"),
        callbacks=make_callback(),
    )

    trainer.fit(model, data_module)
    trainer.test(datamodule=data_module)

    wandb.finish()
