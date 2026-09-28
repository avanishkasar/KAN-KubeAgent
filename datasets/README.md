# Datasets Guide — KAN-KubeAgent

## Overview

This project needs two kinds of data:
1. **Decision-point data to train the KAN gate** — labeled continue/adjust-LR/early-stop points from training runs
2. **Agent evaluation data** — realistic `TrainJob` scenarios for end-to-end testing

---

## Primary Source: Real Training Runs (Fashion-MNIST / small transformer)

**Status:** Generated as part of this research, from real Kubeflow `TrainJob` runs on Minikube.

### Generation Method

```bash
# Step 1: Spin up a local cluster with Kubeflow Trainer installed
minikube start --cpus=4 --memory=8g
kubectl apply -f https://github.com/kubeflow/trainer/releases/download/v2.x/manifests.yaml

# Step 2: Submit real training runs (varying seed / LR schedule / injected bad runs)
python scripts/run_trainjob.py --config configs/fashion_mnist_cnn.yaml --seed 1
python scripts/run_trainjob.py --config configs/fashion_mnist_cnn.yaml --seed 2 --bad-run

# Step 3: Extract feature/decision points from the logged loss history
python scripts/extract_decision_points.py --job finetune-run-01

# Step 4: Weak-label with the reference policy, then manually review a sample
python scripts/label_with_reference_policy.py
```

### Feature → Label Schema

| Feature | Description |
|---------|-------------|
| `loss_plateau_score` | 0-1: how flat the loss has been over the last K epochs |
| `gradient_trend` | direction/magnitude of recent gradient norm change |
| `lr_decay_benefit` | estimated improvement from halving LR now |
| `gpu_hours_remaining_vs_budget` | 0-1: fraction of GPU-hour budget left |
| `epochs_since_improvement` | normalised epochs since best validation loss |

| Label | Meaning |
|-------|---------|
| `continue` | keep training as-is |
| `adjust_lr` | reduce learning rate |
| `early_stop` | stop and free the GPU |

---

## Secondary Source: Synthetic Loss-Curve Generator (toggle, not default)

**Status:** Implemented as a debugging/testing aid, gated behind a `--synthetic` flag.

- Generates curves as exponential decay + Gaussian noise, with an option to inject a plateau at a chosen epoch
- Used for fast KAN-gate unit testing before real training infra is available, and for reproducing specific edge cases on demand
- Never used as the primary training signal for the KAN gate — see `research/proposal/methodology_draft.md` Section 6

```bash
python scripts/generate_synthetic_curve.py --plateau-at-epoch 15 --noise 0.02
```

---

## Reference Baselines (Public — for comparison, not KAN training)

Used only to compare KAN-gated decisions against, per `research/proposal/research_gap.md` Section 4:

| Baseline | Where to get it |
|----------|-------------------|
| Hyperband | [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) implementation |
| ASHA | [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) implementation |

---

## Feature Extraction Schema

```python
FEATURE_SCHEMA = {
    "loss_plateau_score": float,            # 0.0 - 1.0
    "gradient_trend": float,                 # normalised gradient-norm slope
    "lr_decay_benefit": float,               # 0.0 - 1.0
    "gpu_hours_remaining_vs_budget": float,  # 0.0 - 1.0
    "epochs_since_improvement": float        # normalised
}

LABEL_SCHEMA = {
    "decision": str  # "continue" | "adjust_lr" | "early_stop"
}
```
