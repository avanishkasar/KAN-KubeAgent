"""Hindsight (outcome-based) supervision for the KAN gate.

The original gate was trained on labels from a three-rule hand-written
reference policy, so it could at best become a smooth copy of that rule.
Harvesting the gate's *own* live decisions as new labels is no better:
retraining on them is self-distillation and adds no information.

Hindsight labels come from what actually happened next instead. At a
decision epoch t of a completed run we know the monitored (validation)
loss for the whole budget, so we can measure how much improvement was
still to come if training had simply continued:

    remaining_gain(t) = (best_so_far(t) - best_over_budget) / best_so_far(t)

and turn it into the 0-100 stop score the gate regresses:

    stop_score(t) = 100 * clip(1 - remaining_gain(t) / tolerance, 0, 1)

A run with >= `tolerance` relative improvement still ahead scores 0 (keep
going); one that will never beat its current best scores 100 (stop now).
With the gate's decision bands (continue < 40 <= adjust_lr < 75 <= stop)
and the default 5% tolerance, "early_stop" means "less than 1.25% relative
improvement is left in the rest of the budget".

The label uses the future only at training time; at decision time the gate
still sees nothing but the causal features in kan_gate/features.py.
"""
from __future__ import annotations

from kan_gate.features import TrainingHistory, extract_features

DEFAULT_TOLERANCE = 0.05
DEFAULT_CHECK_EVERY = 2
DEFAULT_FIRST_CHECK = 4


def remaining_gain(monitored: list[float], epochs_seen: int) -> float:
    """Relative improvement of the best monitored loss still to come after
    `epochs_seen` epochs, if training continued to the end of `monitored`."""
    best_so_far = min(monitored[:epochs_seen])
    best_final = min(monitored)
    return max(0.0, best_so_far - best_final) / max(abs(best_so_far), 1e-8)


def hindsight_score(monitored: list[float], epochs_seen: int,
                    tolerance: float = DEFAULT_TOLERANCE) -> float:
    gain = remaining_gain(monitored, epochs_seen)
    return 100.0 * max(0.0, min(1.0, 1.0 - gain / tolerance))


def decision_points(monitored: list[float], grad_norms: list[float],
                    check_every: int = DEFAULT_CHECK_EVERY,
                    first_check: int = DEFAULT_FIRST_CHECK,
                    tolerance: float = DEFAULT_TOLERANCE,
                    gpu_hours_budget: float = 10.0) -> list[dict]:
    """Every decision point of one completed run, with causal features and
    its hindsight target. Compute budget is accounted per epoch, matching
    agents/local_process_client.py."""
    total = len(monitored)
    points = []
    for seen in range(first_check, total, check_every):
        history = TrainingHistory(
            loss_history=monitored[:seen],
            grad_norm_history=grad_norms[:seen],
            gpu_hours_used=gpu_hours_budget * seen / total,
            gpu_hours_budget=gpu_hours_budget,
        )
        points.append({
            "epochs_seen": seen,
            "features": extract_features(history),
            "target": hindsight_score(monitored, seen, tolerance),
            "remaining_gain": remaining_gain(monitored, seen),
        })
    return points
