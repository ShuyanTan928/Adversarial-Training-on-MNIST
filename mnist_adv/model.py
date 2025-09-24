"""Model definitions for MNIST adversarial training."""
from __future__ import annotations

import torch
from torch import nn


class MNISTConvNet(nn.Module):
    """Canonical convolutional network used throughout the experiments.

    The architecture follows the layout provided in the project description::

        Conv(32, 5×5) → ReLU → MaxPool(2×2) →
        Conv(64, 5×5) → ReLU → MaxPool(2×2) →
        FC(1024) → ReLU → Dropout(p=0.5) → FC(10).
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10),
        )

        # Xavier/He initialisation is a sensible default for convolutional nets.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
