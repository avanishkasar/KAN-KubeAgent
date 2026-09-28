"""Dashboard backend: runs the real agent loop against a MockTrainJobClient
and serves the results as JSON for dashboard/frontend/index.html.

This is not mock/fake dashboard data - every field in the response comes
from actually invoking agents.graph.build_graph() and kan_gate.gate.KANGate,
the same pipeline agents/run_loop.py exercises on the CLI. Swapping in
KubeflowTrainJobClient for a real cluster later requires no change here -
see agents/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agents.graph import build_graph  # noqa: E402
from agents.trainjob_client import MockTrainJobClient  # noqa: E402
from kan_gate.gate import DEFAULT_CKPT_PATH, KANGate  # noqa: E402

app = FastAPI(title="KAN-KubeAgent Dashboard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_gate_cache: KANGate | None = None


def get_gate() -> KANGate:
    global _gate_cache
    if _gate_cache is None:
        try:
            _gate_cache = KANGate.load(DEFAULT_CKPT_PATH)
        except Exception:
            _gate_cache = KANGate()
    return _gate_cache


class RunRequest(BaseModel):
    job_name: str = "finetune-demo-01"
    epochs: int = 30
    plateau_at_epoch: int | None = 12
    check_every: int = 3
    seed: int = 7
    synthetic: bool = True  # kept explicit: the toggle from methodology_draft.md 6.2


@app.post("/api/runs")
def run_pipeline(req: RunRequest):
    client = MockTrainJobClient()
    client.create(req.job_name, total_epochs=req.epochs,
                   plateau_at_epoch=req.plateau_at_epoch, seed=req.seed)
    gate = get_gate()
    graph = build_graph(client, gate)

    state = {"trainjob_name": req.job_name, "audit_log": []}
    running = True
    while running:
        running = client.step(req.job_name)
        status = client.get_status(req.job_name)
        if status.epoch % req.check_every != 0 and running:
            continue
        state = graph.invoke(state)
        if status.phase != "Running":
            break

    final_status = client.get_status(req.job_name)
    return {
        "job_name": req.job_name,
        "synthetic": req.synthetic,
        "phase": final_status.phase,
        "final_epoch": final_status.epoch,
        "total_epochs": final_status.total_epochs,
        "loss_history": final_status.loss_history,
        "gpu_hours_used": final_status.gpu_hours_used,
        "gpu_hours_budget": final_status.gpu_hours_budget,
        "audit_log": state["audit_log"],
    }


@app.get("/api/gate")
def gate_info():
    gate = get_gate()
    return {"formula": gate.formula(), "feature_names": [
        "loss_plateau_score", "gradient_trend", "lr_decay_benefit",
        "gpu_hours_remaining_vs_budget", "epochs_since_improvement",
    ]}


frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
