"""Training routines for the MNIST adversarial training assignment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .attacks import fgsm, trades_adv_example
from .metrics import TrainingLogger
from .model import MNISTConvNet
from .utils import accuracy, ensure_dir, save_json

TrainingMethod = Literal["standard", "fgsm", "trades"]


@dataclass
class OptimConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0


@dataclass
class TrainingConfig:
    method: TrainingMethod = "standard"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps: int = 100_000
    epochs: int = 50
    fgsm_epsilon: float = 0.3
    trades_epsilon: float = 0.3
    trades_step_size: float = 0.01
    trades_steps: int = 10
    trades_beta: float = 6.0
    log_every: int = 100
    eval_every: int = 1
    output_dir: Path = Path("results")
    seed: int = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(device: str | torch.device) -> MNISTConvNet:
    model = MNISTConvNet()
    model.to(device)
    return model


def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    epsilons: Iterable[float] = (0.0, 0.1, 0.2, 0.3),
) -> Dict[float, float]:
    """Compute clean and adversarial accuracies for ``epsilons``."""

    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    results: Dict[float, float] = {}

    for epsilon in epsilons:
        correct = 0
        count = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if epsilon == 0:
                adv_inputs = inputs.detach()
            else:
                inputs_adv = inputs.clone().detach().requires_grad_(True)
                model.zero_grad(set_to_none=True)
                logits = model(inputs_adv)
                loss = loss_fn(logits, targets)
                grad = torch.autograd.grad(loss, inputs_adv)[0]
                adv_inputs = torch.clamp(inputs_adv + epsilon * torch.sign(grad), 0.0, 1.0)
                adv_inputs = adv_inputs.detach()

            with torch.no_grad():
                logits_adv = model(adv_inputs)
                preds = logits_adv.argmax(dim=1)

            correct += (preds == targets).sum().item()
            count += targets.size(0)

        results[epsilon] = correct / max(count, 1)

    return results


def _train_step_standard(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str | torch.device,
) -> tuple[float, float]:
    model.train()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()

    return loss.item(), accuracy(logits.detach(), targets)


def _train_step_fgsm(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str | torch.device,
    epsilon: float,
) -> tuple[float, float]:
    model.train()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    adv_inputs = fgsm(model, inputs, targets, epsilon, loss_fn)

    optimizer.zero_grad(set_to_none=True)
    logits = model(adv_inputs)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()

    return loss.item(), accuracy(logits.detach(), targets)


def _train_step_trades(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str | torch.device,
    epsilon: float,
    step_size: float,
    steps: int,
    beta: float,
) -> tuple[float, float]:
    model.train()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    adv_inputs = trades_adv_example(model, inputs, step_size, epsilon, steps)
    model.train()

    optimizer.zero_grad(set_to_none=True)
    logits_clean = model(inputs)
    logits_adv = model(adv_inputs)

    loss_natural = loss_fn(logits_clean, targets)
    loss_robust = nn.functional.kl_div(
        nn.functional.log_softmax(logits_adv, dim=1),
        nn.functional.softmax(logits_clean.detach(), dim=1),
        reduction="batchmean",
    )
    loss = loss_natural + beta * loss_robust
    loss.backward()
    optimizer.step()

    return loss.item(), accuracy(logits_clean.detach(), targets)


def train(
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainingConfig,
    optim_config: OptimConfig | None = None,
    output_path: Optional[Path] = None,
) -> Dict:
    """Run training according to ``config`` and return a metrics summary."""

    optim_config = optim_config or OptimConfig()
    set_seed(config.seed)

    device = torch.device(config.device)
    model = create_model(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=optim_config.learning_rate, weight_decay=optim_config.weight_decay)

    logger = TrainingLogger()
    global_step = 0

    for epoch in range(config.epochs):
        if global_step >= config.max_steps:
            break

        for batch in train_loader:
            if global_step >= config.max_steps:
                break

            if config.method == "standard":
                loss, acc = _train_step_standard(model, batch, optimizer, loss_fn, device)
            elif config.method == "fgsm":
                loss, acc = _train_step_fgsm(model, batch, optimizer, loss_fn, device, config.fgsm_epsilon)
            elif config.method == "trades":
                loss, acc = _train_step_trades(
                    model,
                    batch,
                    optimizer,
                    loss_fn,
                    device,
                    config.trades_epsilon,
                    config.trades_step_size,
                    config.trades_steps,
                    config.trades_beta,
                )
            else:
                raise ValueError(f"Unsupported training method: {config.method}")

            logger.log(loss, acc)
            global_step += 1

        if epoch % config.eval_every == 0:
            pass  # Hooks for future extensions.

    eval_results = evaluate_accuracy(model, test_loader, device)

    metrics = {
        "config": config.__dict__,
        "optim_config": optim_config.__dict__,
        "training": logger.as_dict(),
        "eval": eval_results,
    }

    output_path = output_path or config.output_dir / config.method / "metrics.json"
    ensure_dir(output_path.parent)
    save_json(metrics, output_path)

    ensure_dir((config.output_dir / config.method).parent)
    torch.save(model.state_dict(), config.output_dir / config.method / "model.pt")
    save_json({"robust_accuracy": eval_results}, config.output_dir / config.method / "eval.json")

    return metrics
