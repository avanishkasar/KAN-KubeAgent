# Datasets Guide — KAN-KubeAgent

## Overview

This project needs two kinds of data:
1. **Decision-point data to train the KAN gate** — labeled continue/adjust-LR/early-stop points from training runs
2. **Agent evaluation data** — realistic `TrainJob` scenarios for end-to-end testing

---

## Primary Source: Real Training Runs (Fashion-MNIST)

**Status:** Generated automatically every time a real run happens - either
the dashboard's Live mode (`dashboard/backend/live.py`, backed by a real
local subprocess) or a real Kubeflow `TrainJob` on Minikube once
`agents/kubeflow_client.py` is validated against a live cluster.

### Generation Method

No separate extraction step to run manually - `kan_gate/real_run_logger.py`
saves every real run's decision points (the 5 features + the KAN gate's
decision at each check-in) the moment the run ends, whether it completes
naturally or you hit Stop:

```bash
# 1. Start the dashboard and run one or more real training sessions from
#    the Live tab (see dashboard/README.md) - each ending run writes a file to:
ls kan_gate/data/real_runs/
#   e.g. live-finetune-01_1790659331.json

# 2. Once you have a handful of runs accumulated, retrain the gate on them
#    (this is the DEFAULT mode - no --synthetic flag):
python -m kan_gate.train --real-data-dir kan_gate/data/real_runs
```

`training/fashion_mnist_cnn.py` has no synthetic-data fallback by design:
if Fashion-MNIST can't be downloaded, the run fails loudly instead of
silently substituting fake data, so every decision point saved here came
from a genuine training run.

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

**Status:** Implemented as a debugging/testing aid (`kan_gate/synthetic.py`), gated behind a `--synthetic` flag on `kan_gate/train.py` and behind the dashboard's separate Mock/Synthetic tab (`dashboard/frontend/index.html`) - the Live tab and its real training runs never touch this.

- Generates curves as exponential decay + Gaussian noise, with an option to inject a plateau at a chosen epoch
- Used for fast KAN-gate unit testing before real training infra is available, and for reproducing specific edge cases on demand
- Never used as the primary training signal for the KAN gate — see `research/proposal/methodology_draft.md` Section 6

```bash
python -m kan_gate.train --synthetic --steps 250
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
