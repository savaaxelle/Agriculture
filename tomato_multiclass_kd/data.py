from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
)


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_loaders(data_root, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    root = Path(data_root)

    train_ds = datasets.ImageFolder(root / "train", transform=train_transform())
    val_ds = datasets.ImageFolder(root / "val", transform=eval_transform())
    test_ds = datasets.ImageFolder(root / "test", transform=eval_transform())

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError("train/val/test class folders do not match.")

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        ),
    }

    return loaders, train_ds.classes
