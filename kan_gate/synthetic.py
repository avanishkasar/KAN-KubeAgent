"""Synthetic loss-curve generator.

Debugging/testing aid only - per research/proposal/methodology_draft.md
Section 6.2, this is never the KAN gate's primary data source. Real
training runs (kan_gate/train.py + agents/) are the primary path; this
module exists behind an explicit --synthetic flag / dashboard toggle so
the gate can be tested and edge cases reproduced without a live cluster.
"""
from __future__ import annotations

import random

from kan_gate.features import TrainingHistory


def generate_synthetic_history(
    num_epochs: int = 30,
    plateau_at_epoch: int | None = None,
    noise: float = 0.02,
    initial_loss: float = 2.5,
    decay_rate: float = 0.08,
    gpu_hours_budget: float = 10.0,
    hours_per_epoch: float = 0.3,
    seed: int | None = None,
) -> TrainingHistory:
    """Generate a fake but plausible loss/gradient curve.

    Before `plateau_at_epoch`, loss follows exponential decay with noise.
    From that epoch on, it flattens out (simulating a converged/stuck run).
    """
    rng = random.Random(seed)
    losses: list[float] = []
    grad_norms: list[float] = []

    plateau_value = None
    for epoch in range(num_epochs):
        if plateau_at_epoch is not None and epoch >= plateau_at_epoch:
            if plateau_value is None:
                plateau_value = losses[-1]
            loss = plateau_value + rng.gauss(0, noise * 0.3)
            grad = rng.uniform(0.0, 0.05)
        else:
            loss = initial_loss * (2.71828 ** (-decay_rate * epoch)) + rng.gauss(0, noise)
            grad = max(0.05, 1.0 - epoch / max(1, num_epochs))

        losses.append(max(0.01, loss))
        grad_norms.append(max(0.0, grad))

    return TrainingHistory(
        loss_history=losses,
        grad_norm_history=grad_norms,
        lr=2e-4,
        gpu_hours_used=num_epochs * hours_per_epoch,
        gpu_hours_budget=gpu_hours_budget,
    )
