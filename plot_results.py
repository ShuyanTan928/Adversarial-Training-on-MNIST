"""Utility for visualising clean and robust accuracies across training methods."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from mnist_adv.utils import load_json


def load_eval(path: Path) -> Dict[float, float]:
    data = load_json(path)
    return {float(k): v for k, v in data["robust_accuracy"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="List of eval.json files produced by the training script.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/robust_accuracy.png"))
    args = parser.parse_args()

    accuracies: Dict[str, List[float]] = {}
    epsilons = None
    for path in args.paths:
        metrics = load_eval(path)
        if epsilons is None:
            epsilons = sorted(metrics.keys())
        method = path.parent.name
        accuracies[method] = [metrics[e] for e in epsilons]

    if epsilons is None:
        raise ValueError("No evaluation files provided.")

    plt.figure(figsize=(6, 4))
    for method, values in sorted(accuracies.items()):
        plt.plot(epsilons, values, marker="o", label=method.title())

    plt.xlabel("FGSM $\\epsilon$")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
