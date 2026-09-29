"""Persists decision points from a completed real run (live dashboard or
agents/run_loop.py) into the format kan_gate.train.load_real_dataset reads.

This is the harvesting half of the loop research/proposal/methodology_draft.md
Section 6.1 describes: run real training under the agent+gate loop, save
what features led to what decision, then retrain the gate on accumulated
real runs instead of synthetic ones (`python -m kan_gate.train`, no
--synthetic - that flag stays a testing-only aid, per the project's policy
that nothing shown as a result is trained on fake data).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

DEFAULT_REAL_RUNS_DIR = Path(__file__).parent / "data" / "real_runs"


def save_decision_points(job_name: str, audit_log: list[dict],
                          out_dir: Path = DEFAULT_REAL_RUNS_DIR) -> Path | None:
    """Writes one JSON file of {"features", "decision"} pairs for this run.

    Returns the written path, or None if the run produced no decision
    points worth saving (e.g. audit_log entries predating this feature,
    or an empty run).
    """
    entries = [
        {"features": e["features"], "decision": e["decision"]}
        for e in audit_log if "features" in e
    ]
    if not entries:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_name)
    path = out_dir / f"{safe_name}_{int(time.time())}.json"
    path.write_text(json.dumps(entries, indent=2))
    return path
