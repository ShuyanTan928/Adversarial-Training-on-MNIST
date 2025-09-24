"""Data loading utilities for MNIST adversarial training experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    """Configuration options for the MNIST data pipeline."""

    root: str = "data"
    batch_size: int = 50
    num_workers: int = 2
    download: bool = True


def get_transforms() -> transforms.Compose:
    """Return the canonical MNIST preprocessing pipeline.

    The instructions require the inputs to be scaled to ``[0, 1]``.  The
    :class:`torchvision.transforms.ToTensor` operator performs exactly this
    scaling by converting the incoming PIL image to ``float32`` and dividing by
    255.
    """

    return transforms.Compose([transforms.ToTensor()])


def get_dataloaders(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """Create the MNIST train and test :class:`~torch.utils.data.DataLoader`s.

    Parameters
    ----------
    config:
        Collection of data loading options.  The defaults match the
        specification in the assignment prompt.
    """

    transform = get_transforms()

    train_dataset = datasets.MNIST(
        root=config.root,
        train=True,
        transform=transform,
        download=config.download,
    )
    test_dataset = datasets.MNIST(
        root=config.root,
        train=False,
        transform=transform,
        download=config.download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
