"""Utility helpers for tracking metrics during training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RunningAverage:
    """Maintains a numerically-stable running average."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


@dataclass
class TrainingLogger:
    """Accumulates per-iteration and per-epoch summaries."""

    history: Dict[str, List[float]] = field(default_factory=lambda: {"loss": [], "accuracy": []})

    def log(self, loss: float, accuracy: float) -> None:
        self.history["loss"].append(loss)
        self.history["accuracy"].append(accuracy)

    def as_dict(self) -> Dict[str, List[float]]:
        return self.history
