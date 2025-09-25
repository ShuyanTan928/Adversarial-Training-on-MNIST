from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from mnist_adv.utils import load_json


def load_eval(path: Path) -> Dict[float, float]:
    """Loads robust accuracy data from an eval.json file."""
    data = load_json(path)
    return {float(k): v for k, v in data["robust_accuracy"].items()}

def plot_single_result(eval_path: Path, output_dir: Path):
    """
    Loads data from a single eval.json, plots it, and saves the figure.
    """
    metrics = load_eval(eval_path)
    epsilons = sorted(metrics.keys())
    accuracies = [metrics[e] for e in epsilons]

    method = eval_path.parent.name

    plt.figure(figsize=(6, 4))
    plt.plot(epsilons, accuracies, marker="o", label=method.title())

    plt.title(f"Robustness of '{method.title()}' Model")
    plt.xlabel("FGSM $\\epsilon$")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()


    output_path = output_dir / f"robust_accuracy_{method}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close() #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results"),
        help="Root directory containing the training results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where the output plots will be saved.",
    )
    args = parser.parse_args()


    eval_files = list(args.input_dir.rglob("eval.json"))

    if not eval_files:
        print(f"Error: No 'eval.json' files found in '{args.input_dir}'.")
        return

    print(f"Found {len(eval_files)} result files. Generating plots...")
    for eval_path in sorted(eval_files):
        plot_single_result(eval_path, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()