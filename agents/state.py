"""Shared state threaded through the LangGraph agent loop."""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    trainjob_name: str

    # Metrics Watcher output
    loss_history: list[float]
    grad_norm_history: list[float]
    lr: float
    gpu_hours_used: float
    gpu_hours_budget: float

    # Loss-Curve Analyst + Cost Estimator output
    features: dict[str, float]

    # Supervisor's proposal (LLM reasoning, before the KAN gate is consulted)
    proposed_action: str          # "continue" | "adjust_lr" | "early_stop"
    proposal_rationale: str
    proposed_new_lr: float

    # KAN gate output (authoritative - see agents/README.md)
    gate_decision: str
    gate_score: float
    gate_formula: str

    # Executor output
    executed: bool
    audit_log: list[dict]
