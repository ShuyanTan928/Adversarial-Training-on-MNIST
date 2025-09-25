# Adversarial Training on MNIST

This repository provides a light-weight, fully reproducible implementation of
three training regimes for MNIST classification:

* **Standard training** (natural images only).
* **FGSM adversarial training** using a single-step ℓ∞ perturbation.
* **TRADES adversarial training** with a KL-divergence based robustness term.

The code follows the specification in the assignment handout: data is drawn from
MNIST with pixel intensities scaled to ``[0, 1]`` via ``transforms.ToTensor`` and
models use the standard convolutional network architecture described in the
prompt.  Robustness is evaluated against the FGSM attack for
``ε ∈ {0, 0.1, 0.2, 0.3}``.

## Environment setup

1. Create and activate a Python 3.9+ environment.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running experiments

The central entry point is ``train.py``.  The script downloads MNIST if
necessary, trains the requested model, evaluates robust accuracy, and stores
artifacts under ``results/<method>``.

```bash
# Standard training example
python train.py --method standard --epochs 10

# FGSM adversarial training with the default ε = 0.3
python train.py --method fgsm --epochs 10

# TRADES adversarial training (10 inner steps, β = 6)
python train.py --method trades --epochs 10 --trades-steps 10
```

The script emits two JSON summaries per run:

* ``results/<method>/metrics.json`` records the configuration and the full loss
  and accuracy curves.
* ``results/<method>/eval.json`` stores the post-training robust accuracy for
  the evaluation epsilons.

Trained weights are also saved to ``results/<method>/model.pt`` for later use.

## Visualising robustness

After collecting the evaluation results for multiple training regimes you can
plot the accuracy-vs-ε curve:

```bash
python plot_results.py
```

The resulting figure is written to ``results/robust_accuracy_.png`` by default.

## Notes

* Training uses Adam with a learning rate of ``1e-4`` and a batch size of ``50``
  by default, which matches the assignment specification.
* The ``--max-steps`` argument caps the number of optimisation steps to 100,000
  (as required).  The ``--epochs`` parameter provides an additional early stop
  condition when desired.
* Evaluation strictly follows the FGSM protocol from the prompt and clamps the
  perturbed samples to ``[0, 1]``.
