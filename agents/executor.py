"""Executor: applies the KAN gate's decision to a real (or mock) TrainJob.

The gate's decision is authoritative - this module never re-derives or
second-guesses it, it just carries it out and records the audit trail.
See research/proposal/methodology_draft.md Section 5.
"""
from __future__ import annotations

from agents.trainjob_client import TrainJobClient


class KANGatedExecutor:
    def __init__(self, client: TrainJobClient):
        self.client = client

    def execute(self, trainjob_name: str, decision: str, score: float, formula: str,
                new_lr: float | None = None) -> dict:
        entry = {
            "trainjob": trainjob_name,
            "decision": decision,
            "gate_score": score,
            "gate_formula": formula,
            "outcome": None,
        }

        if decision == "continue":
            entry["outcome"] = "NO_OP"

        elif decision == "adjust_lr":
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
