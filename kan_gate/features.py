"""Feature extraction for the KAN gate.

Turns a TrainJob's recent training history into the 5-dimensional feature
vector the KAN gate scores. See research/proposal/methodology_draft.md
Section 2.2 for the feature definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FEATURE_NAMES = [
    "loss_plateau_score",
    "gradient_trend",
    "lr_decay_benefit",
    "gpu_hours_remaining_vs_budget",
    "epochs_since_improvement",
]


@dataclass
class TrainingHistory:
    """Minimal view of a TrainJob's progress needed to score a decision.

    `loss_history` is the *monitored* loss: validation loss when the job
    reports it, training loss otherwise (see TrainJobStatus.to_history).
    """

    loss_history: list[float]
    grad_norm_history: list[float] = field(default_factory=list)
    lr: float = 1e-3
    gpu_hours_used: float = 0.0
    gpu_hours_budget: float = 10.0
    window: int = 5


def _plateau_score(losses: list[float], window: int) -> float:
    """0 (still improving fast) .. 1 (flat) based on relative loss change."""
    if len(losses) < 2:
        return 0.0
    recent = losses[-window:]
    if len(recent) < 2:
        return 0.0
    total_drop = recent[0] - recent[-1]
    scale = abs(recent[0]) + 1e-8
    relative_drop = max(total_drop, 0.0) / scale
    # Small relative drop over the window => high plateau score.
    return max(0.0, min(1.0, 1.0 - relative_drop * 10))


def _gradient_trend(grad_norms: list[float], window: int) -> float:
    """0 (gradient shrunk to nothing) .. 1 (as large as it has ever been).

    Recent mean gradient norm relative to the largest seen so far in this
    run. Normalising by the run's own peak keeps the feature scale-free:
    raw gradient norms differ by orders of magnitude between models (a
    small CNN's global norm is typically 2-5, which an absolute clip to
    [0, 1] would saturate at 1 for the whole run).
    """
    if not grad_norms:
        return 0.0
    peak = max(grad_norms)
    if peak <= 0:
        return 0.0
    recent = grad_norms[-window:]
    avg = sum(recent) / len(recent)
    return max(0.0, min(1.0, avg / peak))


def _lr_decay_benefit(losses: list[float], window: int) -> float:
    """Heuristic 0..1: how much the loss is still oscillating around its
    trend, which a smaller learning rate would typically smooth out."""
    recent = losses[-window:]
    if len(recent) < 3:
        return 0.0
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    sign_changes = sum(
        1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0
    )
    return max(0.0, min(1.0, sign_changes / max(1, len(diffs) - 1)))


def _epochs_since_improvement(losses: list[float], normalise_by: int = 30) -> float:
    if not losses:
        return 0.0
    best = min(losses)
    best_idx = losses.index(best)
    since = (len(losses) - 1) - best_idx
    return max(0.0, min(1.0, since / normalise_by))


def extract_features(history: TrainingHistory) -> dict[str, float]:
    """Compute the 5 KAN-gate features from a TrainingHistory snapshot."""
    budget_used_ratio = (
        history.gpu_hours_used / history.gpu_hours_budget
        if history.gpu_hours_budget > 0
        else 1.0
    )
    remaining_vs_budget = max(0.0, min(1.0, 1.0 - budget_used_ratio))

    return {
        "loss_plateau_score": _plateau_score(history.loss_history, history.window),
        "gradient_trend": _gradient_trend(history.grad_norm_history, history.window),
        "lr_decay_benefit": _lr_decay_benefit(history.loss_history, history.window),
        "gpu_hours_remaining_vs_budget": remaining_vs_budget,
        "epochs_since_improvement": _epochs_since_improvement(history.loss_history),
    }
