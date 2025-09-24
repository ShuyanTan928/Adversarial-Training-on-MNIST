"""Shared helper functions."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import json

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute the top-1 accuracy for the given batch."""

    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())
