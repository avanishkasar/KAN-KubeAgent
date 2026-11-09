"""Closed-loop baseline: standard practice for stopping and learning-rate
control, run on the same live training processes as experiments/closed_loop.py.

Standard practice in PyTorch and Keras is a reduce-on-plateau learning-rate
schedule combined with patience-based early stopping, both on validation
loss. This script applies exactly that policy to the 12 configurations of
one seed of a workload and compares every controlled run with its
uncontrolled recording (same seed, same settings), so its results pair
run by run with the KAN controller's closed-loop results.

Policy, evaluated after every epoch (as ReduceLROnPlateau and
EarlyStopping are):
  * reduce on plateau: if validation loss has not improved by more than
    a relative 1e-4 for `--lr-patience` epochs, halve the learning rate
    (factor 0.5, the same step the KAN controller uses) and reset the
    plateau counter;
  * early stopping: if validation loss has not improved for
    `--stop-patience` epochs, stop the run.

Control runs through the same asynchronous client as the KAN controller:
a learning-rate change takes effect at the next epoch boundary the
process reads, and a stop terminates the process.

A second policy, --policy kanstop, keeps the same reduce-on-plateau
learning-rate schedule but replaces patience-based stopping with the KAN
gate trained on the other three workloads (experiments/results/gates/
kan_lowo_<workload>): at every decision epoch (4, 6, ..., 38) the gate sees
the run's history up to that epoch and the run is stopped if the gate's
decision is early_stop. Its adjust band is ignored, because the learning
rate is left to the plateau schedule.

    python -m experiments.closed_loop_baseline --workload cifar10_cnn
    python -m experiments.closed_loop_baseline --workload cifar10_cnn --policy kanstop
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import agents.local_process_client as lpc
from experiments.collect_curves import WORKLOADS, curves_path
from kan_gate.features import TrainingHistory, extract_features
from kan_gate.gate import KANGate
from kan_gate.hindsight import DEFAULT_CHECK_EVERY, DEFAULT_FIRST_CHECK

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"

# Same training-side settings as experiments/closed_loop.py.
lpc._MODE_TRAINING_ARGS["normal"] = {"workers": 0, "amp": False, "cpu_threads": 2, "throttle_ms": 0}

THRESHOLD = 1e-4   # relative improvement threshold, ReduceLROnPlateau's default ("rel" mode)
FACTOR = 0.5


class PlateauPolicy:
    """ReduceLROnPlateau + EarlyStopping on validation loss, one epoch at a time.
    stop_patience=None disables patience-based stopping."""

    def __init__(self, lr: float, lr_patience: int, stop_patience: int | None):
        self.lr = lr
        self.lr_patience = lr_patience
        self.stop_patience = stop_patience
        self.best = float("inf")
        self.bad_lr = 0       # epochs without improvement since the last LR change
        self.bad_stop = 0     # epochs without improvement (for early stopping)

    def step(self, val_loss: float) -> str:
        if val_loss < self.best * (1 - THRESHOLD):
            self.best = val_loss
            self.bad_lr = 0
            self.bad_stop = 0
        else:
            self.bad_lr += 1
            self.bad_stop += 1
        if self.stop_patience is not None and self.bad_stop >= self.stop_patience:
            return "early_stop"
        if self.bad_lr >= self.lr_patience:
            self.bad_lr = 0
            self.lr *= FACTOR
            return "adjust_lr"
        return "continue"


def gate_stops(gate: KANGate, status, seen: int, epochs: int) -> tuple[bool, float]:
    """KAN gate decision on the history up to epoch `seen` only."""
    history = TrainingHistory(loss_history=status.val_loss_history[:seen],
                              grad_norm_history=status.grad_norm_history[:seen],
                              lr=status.lr, gpu_hours_used=status.gpu_hours_budget * seen / epochs,
                              gpu_hours_budget=status.gpu_hours_budget)
    result = gate.decide(extract_features(history))
    return result.decision == "early_stop", result.score


def control_one(rec: dict, epochs: int, extra_args: list[str], lr_patience: int,
                stop_patience: int | None, gate_path: Path | None = None) -> dict:
    # One gate object per run: pykan models cache activations during a
    # forward pass, so parallel runs must not share one.
    gate = KANGate.load(gate_path) if gate_path is not None else None
    client = lpc.LocalProcessTrainJobClient()
    name = f"baseline-{rec['run_id']}"
    client.create(name, epochs=epochs, lr=rec["lr"], batch_size=rec["batch_size"],
                  subset_size=rec["subset_size"], mode="normal", seed=rec["seed"],
                  extra_args=extra_args)
    policy = PlateauPolicy(rec["lr"], lr_patience, stop_patience)
    seen = 0
    decisions = []
    t0 = time.time()
    while True:
        status = client.get_status(name)
        while seen < len(status.val_loss_history) and status.phase == "Running":
            decision = policy.step(status.val_loss_history[seen])
            seen += 1
            if (gate is not None and seen >= DEFAULT_FIRST_CHECK
                    and (seen - DEFAULT_FIRST_CHECK) % DEFAULT_CHECK_EVERY == 0):
                stop, score = gate_stops(gate, status, seen, epochs)
                if stop:
                    decision = "early_stop"
                decisions.append({"after_epoch": seen, "decision": "gate", "gate_score": score})
            if decision == "adjust_lr":
                client.patch_lr(name, policy.lr)
                decisions.append({"after_epoch": seen, "decision": decision, "new_lr": policy.lr})
            elif decision == "early_stop":
                client.stop(name)
                decisions.append({"after_epoch": seen, "decision": decision})
                break
        if status.phase != "Running" or (decisions and decisions[-1]["decision"] == "early_stop"):
            break
        time.sleep(0.05)
    while client.get_status(name).phase == "Running":
        time.sleep(0.1)
    status = client.get_status(name)
    return {
        "run_id": rec["run_id"], "lr": rec["lr"], "batch_size": rec["batch_size"], "seed": rec["seed"],
        "phase": status.phase, "epochs_run": status.epoch,
        "val_loss": status.val_loss_history, "val_acc": status.val_acc_history,
        "workload": rec.get("workload", "fmnist_cnn"),
        "decisions": decisions,
        "wall_seconds": time.time() - t0,
        "uncontrolled": {"val_loss": rec["val_loss"], "val_acc": rec["val_acc"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="fmnist_cnn")
    parser.add_argument("--test-seed", type=int, default=2)
    parser.add_argument("--policy", choices=["plateau", "kanstop"], default="plateau")
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--stop-patience", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.policy == "kanstop":
        args.out = args.out or RESULTS / f"closed_loop_{args.workload}_kanstop_plateau_lr{args.lr_patience}.json"
        gate = RESULTS / "gates" / f"kan_lowo_{args.workload}"
        stop_patience = None
    else:
        args.out = args.out or RESULTS / (f"closed_loop_{args.workload}_plateau"
                                          f"_lr{args.lr_patience}_stop{args.stop_patience}.json")
        gate, stop_patience = None, args.stop_patience
    extra_args = WORKLOADS[args.workload][0]
    recs = [json.loads(l) for l in curves_path(args.workload).read_text().splitlines() if l.strip()]
    recs = sorted((r for r in recs if r["seed"] == args.test_seed), key=lambda r: (r["lr"], r["batch_size"]))
    epochs = len(recs[0]["val_loss"])
    print(f"{len(recs)} configs, seed {args.test_seed}, policy {args.policy} "
          f"(lr patience {args.lr_patience}, stop patience {stop_patience})", flush=True)

    def job(rec):
        res = control_one(rec, epochs, extra_args, args.lr_patience, stop_patience, gate)
        print(f"{rec['run_id']}: ran {res['epochs_run']}/{epochs} epochs, best val "
              f"{min(res['val_loss']):.4f} (uncontrolled {min(rec['val_loss']):.4f}), "
              f"{sum(d['decision'] == 'adjust_lr' for d in res['decisions'])} LR halvings", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(job, recs))

    summary = []
    for r in results:
        un = r["uncontrolled"]["val_loss"]
        best_un, best_c = min(un), min(r["val_loss"])
        n_ident = 0
        for a, b in zip(r["val_loss"], un):
            if abs(a - b) > 1e-6:
                break
            n_ident += 1
        summary.append({
            "run_id": r["run_id"], "epochs_run": r["epochs_run"], "epochs_budget": epochs,
            "best_val_controlled": best_c, "best_val_uncontrolled": best_un,
            "rel_change_pct": 100 * (best_c - best_un) / best_un,
            "acc_controlled": r["val_acc"][r["val_loss"].index(best_c)],
            "acc_uncontrolled": r["uncontrolled"]["val_acc"][un.index(best_un)],
            "n_adjust_lr": sum(d["decision"] == "adjust_lr" for d in r["decisions"]),
            "stopped_early": r["phase"] == "Stopped",
            "identical_prefix_epochs": n_ident,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"workload": args.workload, "test_seed": args.test_seed,
                                    "policy": ("reduce_on_plateau+kan_gate_stop" if args.policy == "kanstop"
                                               else "reduce_on_plateau+early_stopping"),
                                    "lr_patience": args.lr_patience, "stop_patience": stop_patience,
                                    "factor": FACTOR, "threshold": THRESHOLD,
                                    "runs": results, "summary": summary}, indent=1))
    total = sum(s["epochs_run"] for s in summary)
    print(f"wrote {args.out}: {total}/{epochs * len(summary)} epochs used "
          f"({100 * (1 - total / (epochs * len(summary))):.1f}% saved)")


if __name__ == "__main__":
    main()
