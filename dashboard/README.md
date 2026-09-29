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

The fastest way to get everything up and running is the one-command
startup script at the repo root - it creates a virtualenv if needed,
installs dependencies, trains the gate checkpoint if it's missing, and
launches the server **detached**, so it keeps training in the background
even after you close the terminal or the browser tab:

```bash
./start.sh          # Linux / DGX
```

```powershell
.\start.ps1          # Windows
```

Then open http://localhost:8000 any time — the browser is only a live
viewer, not required for training to keep running. To stop the server:
`kill $(cat dashboard.pid)` (Linux) or `Stop-Process -Id (Get-Content dashboard.pid)` (Windows).

To run it manually instead:

```bash
pip install -r dashboard/backend/requirements.txt -r agents/requirements.txt -r training/requirements.txt -r kan_gate/requirements.txt
python -m kan_gate.train --synthetic --steps 250   # produces the checkpoint the dashboard loads
uvicorn dashboard.backend.app:app --reload --port 8000
```

It defaults to the **Live** tab: fill in the form (job name, epoch count,
learning rate, batch size, how many images to train on, how often the
agent loop checks in), pick a **mode**, and click "Start real training."
Switch to **Mock / Synthetic** for the instant-run mode.

### Modes and auto-continuous

- **Background** — lowers the training subprocess's OS scheduling
  priority (`agents/process_priority.py`) so your foreground apps get the
  CPU first; training gets whatever's left and automatically gets more
  when the machine is idle. Use this to let the system train while you
  use the PC normally.
- **Normal** — default OS priority, no throughput tuning.
- **Turbo** — raises DataLoader worker count and enables mixed-precision
  training for maximum throughput on a CUDA GPU. GPU compute itself has
  no OS-level "nice" (the NVIDIA driver time-slices on its own), so
  turbo's speedup comes from these training-side settings, not priority.
- **Auto-continuous** — when checked, the moment one real run completes
  the dashboard immediately starts a fresh one (new job name/seed)
  instead of stopping, so the system keeps training and harvesting real
  decision points indefinitely. Click Stop to end it.

**Live mode needs real internet access** to download Fashion-MNIST the
first time (cached under `/tmp/data` after that) - there is no
synthetic-data fallback if it can't reach the dataset. If a Live run fails
immediately with a `FATAL: could not load real Fashion-MNIST` line in the
process log, that's this: check network access, or pre-populate the
`--data-root` directory training/fashion_mnist_cnn.py uses.

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

**Verified on real hardware, including a real gotcha:** tested on a
machine with an NVIDIA RTX 4070 Laptop GPU - the GPU tile correctly showed
real live utilization/memory/temperature. But `torch.cuda.is_available()`
was `False` and training ran on CPU anyway, because a plain
`pip install torch` pulls the CPU-only wheel from PyPI by default, GPU or
not. See `training/requirements.txt` for the fix (installing from
PyTorch's CUDA index instead) - `training/fashion_mnist_cnn.py` now also
logs a warning to the live event log if it detects this exact mismatch
(a GPU present via `nvidia-smi`, but no CUDA support in the installed
torch), so it's diagnosable from the dashboard itself rather than a silent
CPU fallback.

## Harvesting real runs to retrain the gate

Every Live run - whether it completes naturally or you click Stop -
automatically saves its decision points (the 5 features plus the KAN
gate's decision at each check-in) to `kan_gate/data/real_runs/*.json` via
`kan_gate/real_run_logger.py`. Nothing manual to run for this; it happens
as part of ending the session. Once a handful of runs have accumulated:

```bash
python -m kan_gate.train --real-data-dir kan_gate/data/real_runs
```

retrains the gate on real curves instead of the synthetic ones the
shipped checkpoint started from, and overwrites the checkpoint the
dashboard loads. See `datasets/README.md` for the full picture.

## Once a real Kubernetes TrainJob is available

`dashboard/backend/live.py` currently drives
`agents.local_process_client.LocalProcessTrainJobClient` (a real local
subprocess, no cluster needed). Swap it for
`agents.kubeflow_client.KubeflowTrainJobClient` to point Live mode at a
real cluster instead - same `TrainJobClient` interface, so
`agents/graph.py` does not change. See `k8s/README.md` and
`agents/README.md` for that client's current (untested-on-a-live-cluster)
status.
