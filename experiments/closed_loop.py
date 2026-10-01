"""Closed-loop experiment: the full agent + KAN-gate system controlling
real training runs, compared with the same runs left uncontrolled.

Offline replay (experiments/benchmark.py) can only score stop/continue.
Here the gate's decisions really execute - "adjust_lr" patches the running
process's learning rate, "early_stop" terminates it - so the effect of LR
adjustment is measured, not assumed. Each controlled run uses the same
seed and training settings as its uncontrolled counterpart in
experiments/data/curves.jsonl, and the gate was trained only on the other
seeds' curves (experiments/results/gates/kan_hindsight_test_seed<S>).

Control is asynchronous, exactly as in the live dashboard: the training
process keeps running while the agents deliberate, so a decision made
after epoch t takes effect from the next epoch boundary the process reads.

    python -m experiments.closed_loop --test-seed 2
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import agents.local_process_client as lpc
from agents.graph import build_graph
from kan_gate.gate import KANGate
from kan_gate.hindsight import DEFAULT_CHECK_EVERY, DEFAULT_FIRST_CHECK

REPO_ROOT = Path(__file__).resolve().parents[1]
CURVES = REPO_ROOT / "experiments" / "data" / "curves.jsonl"
OUT = REPO_ROOT / "experiments" / "results" / "closed_loop.json"

# Same training-side settings experiments/collect_curves.py used (2 parallel
# runs x 2 threads, in-process data loading) so controlled and uncontrolled
# runs of one seed are directly comparable.
lpc._MODE_TRAINING_ARGS["normal"] = {"workers": 0, "amp": False, "cpu_threads": 2, "throttle_ms": 0}


def control_one(rec: dict, gate_path: Path, epochs: int) -> dict:
    gate = KANGate.load(gate_path)
    client = lpc.LocalProcessTrainJobClient()
    name = f"closed-{rec['run_id']}"
    client.create(name, epochs=epochs, lr=rec["lr"], batch_size=rec["batch_size"],
                  subset_size=rec["subset_size"], mode="normal", seed=rec["seed"])
    graph = build_graph(client, gate)
    state = {"trainjob_name": name, "audit_log": []}
    next_check = DEFAULT_FIRST_CHECK
    t0 = time.time()
    while True:
        status = client.get_status(name)
        if status.epoch >= next_check and status.phase == "Running":
            state = graph.invoke(state)
            entry = state["audit_log"][-1]
            entry["decided_after_epoch"] = status.epoch
            next_check = status.epoch + DEFAULT_CHECK_EVERY
        if status.phase != "Running":
            break
        time.sleep(0.2)
    status = client.get_status(name)
    events = client.get_new_events(name)
    return {
        "run_id": rec["run_id"], "lr": rec["lr"], "batch_size": rec["batch_size"], "seed": rec["seed"],
        "phase": status.phase, "epochs_run": status.epoch,
        "val_loss": status.val_loss_history, "val_acc": status.val_acc_history,
        "decisions": [{k: e.get(k) for k in ("decided_after_epoch", "decision", "gate_score", "outcome",
                                              "new_lr", "llm_proposal")} for e in state["audit_log"]],
        "lr_patch_events": [e for e in events if e.startswith("Learning rate patched")],
        "wall_seconds": time.time() - t0,
        "uncontrolled": {"val_loss": rec["val_loss"], "val_acc": rec["val_acc"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-seed", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()

    gate_path = REPO_ROOT / "experiments" / "results" / "gates" / f"kan_hindsight_test_seed{args.test_seed}"
    recs = [json.loads(l) for l in CURVES.read_text().splitlines() if l.strip()]
    recs = sorted((r for r in recs if r["seed"] == args.test_seed), key=lambda r: (r["lr"], r["batch_size"]))
    epochs = len(recs[0]["val_loss"])
    print(f"{len(recs)} configs, seed {args.test_seed}, gate {gate_path.name}", flush=True)

    def job(rec):
        res = control_one(rec, gate_path, epochs)
        full_best = min(rec["val_loss"])
        best = min(res["val_loss"]) if res["val_loss"] else float("nan")
        print(f"{rec['run_id']}: ran {res['epochs_run']}/{epochs} epochs, best val {best:.4f} "
              f"(uncontrolled {full_best:.4f}), decisions "
              f"{[d['decision'] for d in res['decisions']]}", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(job, recs))

    summary = []
    for r in results:
        un = r["uncontrolled"]["val_loss"]
        best_un = min(un)
        best_c = min(r["val_loss"])
        acc_c = r["val_acc"][r["val_loss"].index(best_c)]
        acc_un = r["uncontrolled"]["val_acc"][un.index(best_un)]
        n_ident = 0
        for a, b in zip(r["val_loss"], un):
            if abs(a - b) > 1e-6:
                break
            n_ident += 1
        summary.append({
            "run_id": r["run_id"], "epochs_run": r["epochs_run"], "epochs_budget": epochs,
            "best_val_controlled": best_c, "best_val_uncontrolled": best_un,
            "rel_change_pct": 100 * (best_c - best_un) / best_un,
            "acc_controlled": acc_c, "acc_uncontrolled": acc_un,
            "n_adjust_lr": sum(d["decision"] == "adjust_lr" for d in r["decisions"]),
            "stopped_early": r["phase"] == "Stopped",
            "identical_prefix_epochs": n_ident,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"test_seed": args.test_seed, "runs": results, "summary": summary}, indent=1))
    total = sum(s["epochs_run"] for s in summary)
    print(f"wrote {OUT}: {total}/{epochs * len(summary)} epochs used "
          f"({100 * (1 - total / (epochs * len(summary))):.1f}% saved)")


if __name__ == "__main__":
    main()
