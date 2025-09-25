"""Command line interface for training MNIST adversarial models."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch

from mnist_adv.data import DataConfig, get_dataloaders
from mnist_adv.training import OptimConfig, TrainingConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["standard", "fgsm", "trades"], default="standard")
    parser.add_argument("--data-root", type=str, default="data", help="Directory to store the MNIST dataset")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--fgsm-epsilon", type=float, default=0.3)
    parser.add_argument("--trades-epsilon", type=float, default=0.3)
    parser.add_argument("--trades-step-size", type=float, default=0.01)
    parser.add_argument("--trades-steps", type=int, default=10)
    parser.add_argument("--trades-beta", type=float, default=6.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_config = DataConfig(root=args.data_root, batch_size=args.batch_size)
    train_loader, test_loader = get_dataloaders(data_config)

    train_config = TrainingConfig(
        method=args.method,
        device=args.device,
        max_steps=args.max_steps,
        epochs=args.epochs,
        fgsm_epsilon=args.fgsm_epsilon,
        trades_epsilon=args.trades_epsilon,
        trades_step_size=args.trades_step_size,
        trades_steps=args.trades_steps,
        trades_beta=args.trades_beta,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    optim_config = OptimConfig(learning_rate=args.lr, weight_decay=args.weight_decay)

    metrics = train(train_loader, test_loader, train_config, optim_config)
    print("Training finished.")
    print("Robust accuracy (ε -> accuracy):", metrics["eval"])


if __name__ == "__main__":
    main()
