# dashboard/ — Live Decision Dashboard

A FastAPI backend plus a static frontend with two modes, switched with the
slider at the top of the page:

- **Live** — runs a real local training subprocess
  (`training/fashion_mnist_cnn.py`) on this machine, samples this
  machine's *real* CPU/RAM/GPU load while it trains
  (`dashboard/backend/hardware.py`), and streams every epoch, hardware
  sample, and KAN gate decision to the page over a WebSocket as it
  happens. Nothing here is precomputed or replayed - the loss curve grows
  point by point because the model is actually training right now.
- **Mock / Synthetic** — the original single-request mode: a synthetic
  loss curve (optionally with an injected plateau) is generated instantly
  and scored by the same real KAN gate. No process runs, nothing streams -
  useful for fast iteration and reproducing a specific edge case on
  demand. See `research/proposal/methodology_draft.md` Section 6.2.

## Running it

```bash
pip install -r dashboard/backend/requirements.txt -r agents/requirements.txt -r training/requirements.txt
python -m kan_gate.train --synthetic --steps 250   # produces the checkpoint the dashboard loads
uvicorn dashboard.backend.app:app --reload --port 8000
```

Open http://localhost:8000 — the backend serves the frontend directly, no
separate dev server needed. It defaults to the **Live** tab: fill in the
form (job name, epoch count, learning rate, batch size, how many images to
train on, how often the agent loop checks in) and click "Start real
training." Switch to **Mock / Synthetic** for the instant-run mode.

## What's actually live, and what isn't

- **Loss/gradient numbers, in both modes**: Live mode reads them from the
  real training subprocess's stdout as it prints them, one epoch at a
  time. Mock mode generates a full synthetic curve up front - no live
  numbers there, by design.
- **CPU/RAM/GPU numbers**: only in Live mode, only while a run is active -
  real reads of this machine via `psutil` and (if present) `nvidia-smi`.
  If no NVIDIA GPU is found, the GPU tile honestly reports "No GPU
  detected" rather than making a number up - this is what makes Live mode
  portable across machines (a laptop, a GPU box, this dashboard's dev
  environment) with zero code changes.
- **The KAN gate's decisions and formula**: real in both modes - the same
  trained `KANGate` checkpoint scores real or synthetic features
  identically either way.
- **The process log panel** (Live mode only): every line is either the
  training subprocess's own stdout, a hardware sample, or an agent/gate
  decision - not a scripted demo narration.

## Testing on different hardware (e.g. a college GPU machine)

Live mode auto-detects what's on the machine it runs on - if `nvidia-smi`
is present and reports a GPU, its utilization/memory/temperature show up
live in the GPU tile; if not, the CPU/RAM tiles are still fully live and
the GPU tile just says so. No configuration needed to move between
machines - the same `uvicorn dashboard.backend.app:app` command works
whether there's a GPU or not.

## Once a real Kubernetes TrainJob is available

`dashboard/backend/live.py` currently drives
`agents.local_process_client.LocalProcessTrainJobClient` (a real local
subprocess, no cluster needed). Swap it for
`agents.kubeflow_client.KubeflowTrainJobClient` to point Live mode at a
real cluster instead - same `TrainJobClient` interface, so
`agents/graph.py` does not change. See `k8s/README.md` and
`agents/README.md` for that client's current (untested-on-a-live-cluster)
status.
