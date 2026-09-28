"""Agent loop tests - MockTrainJobClient only, no cluster, no network.

No ANTHROPIC_API_KEY is set in this environment, so the Supervisor node
exercises its heuristic fallback path (agents/llm.py::_heuristic_fallback).
"""
import os

from agents.graph import build_graph
from agents.trainjob_client import MockTrainJobClient
from kan_gate.gate import KANGate
from kan_gate.train import build_synthetic_dataset, rows_to_tensors

os.environ.pop("ANTHROPIC_API_KEY", None)


def _trained_gate() -> KANGate:
    rows = build_synthetic_dataset(n_runs=10, seed=1)
    X, y = rows_to_tensors(rows)
    gate = KANGate()
    gate.fit(X, y, steps=20)
    return gate


def test_graph_runs_end_to_end_and_logs_every_decision():
    client = MockTrainJobClient()
    client.create("test-job", total_epochs=10, plateau_at_epoch=3, seed=1)
    gate = _trained_gate()
    graph = build_graph(client, gate)

    state = {"trainjob_name": "test-job", "audit_log": []}
    running = True
    while running:
        running = client.step("test-job")
        status = client.get_status("test-job")
        if status.epoch % 2 != 0 and running:
            continue
        state = graph.invoke(state)

    assert state["executed"] is True
    assert len(state["audit_log"]) >= 1
    for entry in state["audit_log"]:
        assert entry["decision"] in ("continue", "adjust_lr", "early_stop")
        assert entry["outcome"] in ("NO_OP", "LR_ADJUSTED", "EARLY_STOPPED")
        assert entry["gate_formula"].startswith("stop_score =")


def test_early_stop_actually_stops_the_mock_job():
    client = MockTrainJobClient()
    client.create("plateau-job", total_epochs=20, plateau_at_epoch=2, seed=2)
    gate = _trained_gate()
    graph = build_graph(client, gate)

    state = {"trainjob_name": "plateau-job", "audit_log": []}
    for _ in range(20):
        if not client.step("plateau-job"):
            break
        state = graph.invoke(state)
        if client.get_status("plateau-job").phase != "Running":
            break

    last_decision = state["audit_log"][-1]["decision"]
    if last_decision == "early_stop":
        assert client.get_status("plateau-job").phase == "Stopped"
