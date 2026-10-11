"""Executor: applies the KAN gate's decision to a real (or mock) TrainJob.

The gate's decision is authoritative - this module never re-derives or
second-guesses it, it just carries it out and records the audit trail.
See research/proposal/methodology_draft.md Section 5.
"""
from __future__ import annotations

from agents.trainjob_client import TrainJobClient


# Learning-rate adjustment guard. A plateaued run keeps scoring in the
# adjust_lr band, and without a guard the executor halved the learning rate
# at every decision point (up to 10 times in one closed-loop run, leaving
# the run frozen at ~1/1000 of its learning rate). An adjustment is now
# carried out only if at least COOLDOWN_CHECKS decision points have passed
# since the previous one, so its effect can show in the loss first, and at
# most MAX_LR_ADJUSTMENTS times per job. A deferred adjustment is logged
# with outcome LR_ADJUST_DEFERRED and leaves the job running unchanged.
MAX_LR_ADJUSTMENTS = 3
COOLDOWN_CHECKS = 2


class KANGatedExecutor:
    def __init__(self, client: TrainJobClient, max_lr_adjustments: int | None = MAX_LR_ADJUSTMENTS,
                 cooldown_checks: int = COOLDOWN_CHECKS):
        """max_lr_adjustments=None and cooldown_checks=0 disable the guard
        (the original, unguarded behaviour)."""
        self.client = client
        self.max_lr_adjustments = max_lr_adjustments
        self.cooldown_checks = cooldown_checks
        self._checks: dict[str, int] = {}
        self._adjustments: dict[str, list[int]] = {}

    def _adjustment_allowed(self, trainjob_name: str) -> bool:
        done = self._adjustments.get(trainjob_name, [])
        if self.max_lr_adjustments is not None and len(done) >= self.max_lr_adjustments:
            return False
        return not done or self._checks[trainjob_name] - done[-1] > self.cooldown_checks

    def execute(self, trainjob_name: str, decision: str, score: float, formula: str,
                new_lr: float | None = None) -> dict:
        entry = {
            "trainjob": trainjob_name,
            "decision": decision,
            "gate_score": score,
            "gate_formula": formula,
            "outcome": None,
        }
        check = self._checks.get(trainjob_name, -1) + 1
        self._checks[trainjob_name] = check

        if decision == "continue":
            entry["outcome"] = "NO_OP"

        elif decision == "adjust_lr" and not self._adjustment_allowed(trainjob_name):
            entry["outcome"] = "LR_ADJUST_DEFERRED"

        elif decision == "adjust_lr":
            self._adjustments.setdefault(trainjob_name, []).append(check)
            lr = new_lr or (self.client.get_status(trainjob_name).lr * 0.5)
            self.client.patch_lr(trainjob_name, lr)
            entry["new_lr"] = lr
            entry["outcome"] = "LR_ADJUSTED"

        elif decision == "early_stop":
            self.client.stop(trainjob_name)
            entry["outcome"] = "EARLY_STOPPED"

        else:
            raise ValueError(f"Unknown gate decision: {decision}")

        return entry
