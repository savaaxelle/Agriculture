from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Iterable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import (
    RAW_DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    TRAIN_RATIO,
)

HEALTHY_CLASS = "Tomato___healthy"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.80, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


def binary_label(original_class: str) -> int:
    return 0 if original_class == HEALTHY_CLASS else 1


def collect_samples(split_dir: Path):
    samples = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                samples.append((image_path, binary_label(class_dir.name), class_dir.name))
    return samples


def stratified_train_val_split(samples, train_ratio: float, seed: int):
    """Split each original tomato class separately to preserve all disease types."""
    groups = defaultdict(list)
    for sample in samples:
        groups[sample[2]].append(sample)

    rng = random.Random(seed)
    train_samples = []
    val_samples = []

    for original_class, group in sorted(groups.items()):
        rng.shuffle(group)
        cut = int(len(group) * train_ratio)
        train_samples.extend(group[:cut])
        val_samples.extend(group[cut:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


class BinaryTomatoDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform
        self.targets = [sample[1] for sample in self.samples]
        self.classes = ["healthy", "sick"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, _ = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def create_dataloaders():
    source_train = RAW_DATA_DIR / "train"
    source_test = RAW_DATA_DIR / "val"

    if not source_train.exists() or not source_test.exists():
        raise FileNotFoundError(
            "Dataset is not extracted. Run: python prepare_dataset.py"
        )

    train_transform, eval_transform = get_transforms()

    all_train_samples = collect_samples(source_train)
    train_samples, val_samples = stratified_train_val_split(
        all_train_samples,
        train_ratio=TRAIN_RATIO,
        seed=RANDOM_SEED,
    )
    test_samples = collect_samples(source_test)

    train_ds = BinaryTomatoDataset(train_samples, transform=train_transform)
    val_ds = BinaryTomatoDataset(val_samples, transform=eval_transform)
    test_ds = BinaryTomatoDataset(test_samples, transform=eval_transform)

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
