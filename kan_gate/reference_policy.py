"""Rule-based reference policy.

Used as (a) the weak-label source for the KAN gate's training data, and
(b) a plain baseline to compare the trained KAN gate against. See
research/proposal/methodology_draft.md Section 6.3.
"""
from __future__ import annotations

DECISIONS = ("continue", "adjust_lr", "early_stop")


def reference_decision(features: dict[str, float]) -> str:
    """A simple, explicit rule set - not interpretable-by-formula like the
    KAN gate, but a reasonable ground truth to train/compare against."""
    plateau = features["loss_plateau_score"]
    budget_left = features["gpu_hours_remaining_vs_budget"]
    lr_benefit = features["lr_decay_benefit"]
    since_improve = features["epochs_since_improvement"]

    if plateau > 0.85 and (budget_left < 0.5 or since_improve > 0.4):
        return "early_stop"
    if plateau > 0.55 and lr_benefit > 0.4:
        return "adjust_lr"
    return "continue"


def decision_to_score(decision: str) -> float:
    """Maps a categorical decision to the continuous stop-score scale the
    KAN model is trained to regress (0=keep going, 100=stop now)."""
    return {"continue": 10.0, "adjust_lr": 50.0, "early_stop": 90.0}[decision]
