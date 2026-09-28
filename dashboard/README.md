# dashboard/ — Live Decision Dashboard

A small FastAPI backend that runs the real agent pipeline
(`agents.graph.build_graph`, the same one `agents/run_loop.py` exercises)
against a `MockTrainJobClient`, and a static frontend that renders the loss
curve, decision markers, the current KAN gate formula, and a full audit log
table. No data on this page is fabricated for display - every number comes
from actually invoking the pipeline.

## Running it

```bash
pip install -r dashboard/backend/requirements.txt
python -m kan_gate.train --synthetic --steps 250   # produces the checkpoint the dashboard loads
uvicorn dashboard.backend.app:app --reload --port 8000
```

Open http://localhost:8000 — the backend serves the frontend directly, no
separate dev server needed. Click "Run pipeline" to execute a fresh run
with the parameters in the form (job name, epoch count, when to check in,
and whether to inject a synthetic plateau).

## What "Inject plateau" is

The run always goes through the real pipeline; "Inject plateau" only
controls whether the underlying `MockTrainJobClient`'s synthetic loss curve
is given a plateau at a chosen epoch or left to decay smoothly with no
plateau. This is the synthetic-data toggle from
`research/proposal/methodology_draft.md` Section 6.2 - a testing/demo aid,
kept as an explicit UI control rather than hidden, per the project's
requirement to always have a visible switch back to synthetic data.

## Once a real TrainJob is available

Swap `MockTrainJobClient` for `agents.kubeflow_client.KubeflowTrainJobClient`
in `dashboard/backend/app.py`'s `run_pipeline` - same interface, no other
change needed. See `k8s/README.md` and `agents/README.md`.
