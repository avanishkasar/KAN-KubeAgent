"""Collect real, uncontrolled learning curves for the offline benchmark.

Runs training/fashion_mnist_cnn.py - the exact script the live dashboard
controls - over a grid of learning rates, batch sizes and seeds, with no
intervention, and records every epoch's train loss, gradient norm,
validation loss/accuracy and wall-clock time. These full curves are what
experiments/benchmark.py replays every stopping policy against.

    python -m experiments.collect_curves --epochs 40 --parallel 2
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = REPO_ROOT / "training" / "fashion_mnist_cnn.py"
DEFAULT_OUT = REPO_ROOT / "experiments" / "data" / "curves.jsonl"

LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
BATCH_SIZES = [64, 256]
SEEDS = [0, 1, 2]


def run_one(lr: float, batch_size: int, seed: int, epochs: int, subset: int,
            val_size: int, threads: int) -> dict:
    cmd = [sys.executable, str(TRAINING_SCRIPT), "--epochs", str(epochs), "--lr", str(lr),
           "--batch-size", str(batch_size), "--subset-size", str(subset),
           "--val-size", str(val_size), "--seed", str(seed), "--workers", "0",
           "--cpu-threads", str(threads)]
    start = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    epochs_log = []
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            epochs_log.append(json.loads(line))
    if out.returncode != 0 or len(epochs_log) != epochs:
        raise RuntimeError(f"run lr={lr} bs={batch_size} seed={seed} failed:\n{out.stdout[-2000:]}")
    return {
        "run_id": f"lr{lr:g}_bs{batch_size}_s{seed}",
        "lr": lr, "batch_size": batch_size, "seed": seed,
        "subset_size": subset, "val_size": val_size,
        "train_loss": [e["loss"] for e in epochs_log],
        "grad_norm": [e["grad_norm"] for e in epochs_log],
        "val_loss": [e["val_loss"] for e in epochs_log],
        "val_acc": [e["val_acc"] for e in epochs_log],
        "epoch_seconds": [e["epoch_seconds"] for e in epochs_log],
        "wall_seconds": time.time() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--subset-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["run_id"] for l in args.out.read_text().splitlines() if l.strip()}

    grid = [(lr, bs, s) for lr, bs, s in itertools.product(LRS, BATCH_SIZES, SEEDS)
            if f"lr{lr:g}_bs{bs}_s{s}" not in done]
    print(f"{len(done)} runs already collected, {len(grid)} to go", flush=True)
    threads = max(1, 4 // args.parallel)

    def job(cfg):
        lr, bs, s = cfg
        rec = run_one(lr, bs, s, args.epochs, args.subset_size, args.val_size, threads)
        with args.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        best = min(rec["val_loss"])
        print(f"done {rec['run_id']}: best val_loss={best:.4f} at epoch "
              f"{rec['val_loss'].index(best) + 1}, {rec['wall_seconds']:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(job, grid))


if __name__ == "__main__":
    main()
