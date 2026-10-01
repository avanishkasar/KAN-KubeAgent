"""Persists a finished real run (live dashboard or agents/run_loop.py) so
the KAN gate can later be retrained on it.

Each file stores the run's full monitored-loss and gradient-norm curves
alongside the decisions the gate made. Storing the curves is what makes
the harvest useful: kan_gate/train.py recomputes *hindsight* labels from
what actually happened next (kan_gate/hindsight.py) instead of reusing the
gate's own decisions as labels, which would only teach it to copy itself.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

DEFAULT_REAL_RUNS_DIR = Path(__file__).parent / "data" / "real_runs"


def save_decision_points(job_name: str, audit_log: list[dict],
                         out_dir: Path = DEFAULT_REAL_RUNS_DIR,
                         monitored_loss: list[float] | None = None,
                         grad_norms: list[float] | None = None,
                         phase: str | None = None) -> Path | None:
    """Writes one JSON record for this run. Returns the written path, or
    None if the run produced no decision points worth saving."""
    entries = [
        {"features": e["features"], "decision": e["decision"],
         "gate_score": e.get("gate_score"), "epoch_index": e.get("epoch_index")}
        for e in audit_log if "features" in e
    ]
    if not entries:
        return None

    record = {
        "job_name": job_name,
        "phase": phase,
        "monitored_loss": monitored_loss or [],
        "grad_norm": grad_norms or [],
        "decision_points": entries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_name)
    path = out_dir / f"{safe_name}_{int(time.time())}.json"
    path.write_text(json.dumps(record, indent=2))
    return path
