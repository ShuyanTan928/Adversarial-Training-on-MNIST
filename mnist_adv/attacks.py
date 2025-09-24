"""Adversarial attack utilities."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def fgsm(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    loss_fn: nn.Module | None = None,
    clamp: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Craft Fast Gradient Sign Method adversarial examples.

    Parameters
    ----------
    model:
        The model under attack.
    inputs, targets:
        Mini-batch of examples and labels.  The tensors must already reside on
        the same device as ``model``.
    epsilon:
        Perturbation budget measured in ℓ∞ norm.
    loss_fn:
        Loss to differentiate.  ``nn.CrossEntropyLoss`` is used when ``None``.
    clamp:
        Lower/upper bounds used to clip the result.
    """

    if epsilon == 0:
        return inputs

    loss_fn = loss_fn or nn.CrossEntropyLoss()

    inputs_adv = inputs.clone().detach().requires_grad_(True)

    logits = model(inputs_adv)
    loss = loss_fn(logits, targets)
    grad = torch.autograd.grad(loss, inputs_adv)[0]
    inputs_adv = inputs_adv + epsilon * torch.sign(grad)
    inputs_adv = torch.clamp(inputs_adv, *clamp)
    return inputs_adv.detach()


def clamp_linf(original: torch.Tensor, perturbed: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Project ``perturbed`` into an ℓ∞ ball of radius ``epsilon`` around ``original``."""

    return torch.max(torch.min(perturbed, original + epsilon), original - epsilon)


def trades_adv_example(
    model: nn.Module,
    inputs: torch.Tensor,
    step_size: float,
    epsilon: float,
    steps: int,
) -> torch.Tensor:
    """Generate TRADES-style adversarial samples using projected gradient ascent.

    The optimisation maximises the KL divergence between the predictions on the
    clean and perturbed inputs.  This routine is intentionally general so it can
    be re-used for both training and evaluation.
    """

    if epsilon == 0:
        return inputs

    model.eval()

    inputs_adv = inputs.detach() + 0.001 * torch.randn_like(inputs)
    inputs_adv = torch.clamp(inputs_adv, 0.0, 1.0)
    inputs_adv.requires_grad_(True)

    with torch.no_grad():
        logits_clean = model(inputs)
        probs_clean = nn.functional.softmax(logits_clean, dim=1)

    for _ in range(steps):
        logits_adv = model(inputs_adv)
        log_probs_adv = nn.functional.log_softmax(logits_adv, dim=1)
        loss_kl = nn.functional.kl_div(log_probs_adv, probs_clean, reduction="batchmean")

        grad = torch.autograd.grad(loss_kl, inputs_adv)[0]
        inputs_adv = inputs_adv + step_size * torch.sign(grad)
        inputs_adv = clamp_linf(inputs, inputs_adv, epsilon)
        inputs_adv = torch.clamp(inputs_adv, 0.0, 1.0)
        inputs_adv = inputs_adv.detach()
        inputs_adv.requires_grad_(True)

    return inputs_adv.detach()
