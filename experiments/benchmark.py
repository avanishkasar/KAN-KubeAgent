"""Offline benchmark of stopping policies on real learning curves.

Every policy is replayed against the same complete, uncontrolled curves
from experiments/collect_curves.py (counterfactual replay: a policy that
stops at epoch t is credited with the best validation loss seen up to t).
Evaluation is 3-fold cross-validation grouped by training seed: learned
gates are fit on the curves of two seeds and tested on the third, so no
test run - and no run sharing a test run's weight initialisation - is
ever seen in training.

Only the stop/continue axis can be replayed offline: an "adjust_lr"
decision changes the future curve, which a recorded curve cannot show.
Here it is treated as "continue"; its real effect is measured on live
runs by experiments/closed_loop.py.

    python -m experiments.benchmark
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import sympy
import torch
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from kan_gate.features import FEATURE_NAMES
from kan_gate.gate import ADJUST_LR_MAX, CONTINUE_MAX, KANGate
from kan_gate.hindsight import DEFAULT_CHECK_EVERY, DEFAULT_FIRST_CHECK, decision_points, remaining_gain
from kan_gate.reference_policy import decision_to_score, reference_decision
from kan_gate.train import build_synthetic_dataset, rows_to_tensors

REPO_ROOT = Path(__file__).resolve().parents[1]
CURVES = REPO_ROOT / "experiments" / "data" / "curves.jsonl"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

TAU = 0.05            # hindsight tolerance used for training labels
REGRET_FAIL = 0.02    # a stop that loses >2% of the achievable val loss counts as premature
GATE_SEEDS = [0, 1, 2]
KAN_STEPS = 200


# ----------------------------------------------------------------------------- data

def load_runs(path: Path = CURVES) -> list[dict]:
    runs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in runs:
        r["points"] = decision_points(r["val_loss"], r["grad_norm"], DEFAULT_CHECK_EVERY,
                                      DEFAULT_FIRST_CHECK, TAU)
        r["X"] = np.array([[p["features"][n] for n in FEATURE_NAMES] for p in r["points"]],
                          dtype=np.float32)
    return sorted(runs, key=lambda r: (r["seed"], r["lr"], r["batch_size"]))


def rows_from(runs: list[dict], tau: float) -> list[tuple[dict, float]]:
    rows = []
    for r in runs:
        for p in decision_points(r["val_loss"], r["grad_norm"], DEFAULT_CHECK_EVERY,
                                 DEFAULT_FIRST_CHECK, tau):
            rows.append((p["features"], p["target"]))
    return rows


def banded(scores) -> np.ndarray:
    s = np.asarray(scores)
    return np.where(s < CONTINUE_MAX, 0, np.where(s < ADJUST_LR_MAX, 1, 2))


# ----------------------------------------------------------------------------- scorers

class Scorer:
    name = "?"
    kind = "?"

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class KANScorer(Scorer):
    kind = "kan"

    def __init__(self, gate: KANGate, name: str):
        self.gate, self.name = gate, name

    def predict(self, X):
        with torch.no_grad():
            out = self.gate.model(torch.tensor(X, dtype=torch.float32)).numpy()[:, 0]
        return np.clip(out, 0, 100)


class SkScorer(Scorer):
    def __init__(self, model, name: str, kind: str):
        self.model, self.name, self.kind = model, name, kind

    def predict(self, X):
        return np.clip(self.model.predict(X), 0, 100)


class RuleScorer(Scorer):
    name, kind = "Reference rule (teacher)", "rule"

    def predict(self, X):
        return np.array([decision_to_score(reference_decision(dict(zip(FEATURE_NAMES, x))))
                         for x in X])


def fit_kan(rows, seed: int, name: str) -> tuple[KANScorer, dict]:
    """Train a KAN gate exactly like kan_gate.train.train_gate, but keep the
    spline (pre-symbolic) predictions so the cost of symbolification can be
    measured."""
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(rows), generator=g).tolist()
    rows = [rows[i] for i in order]
    split = int(len(rows) * 0.8)
    X_tr, y_tr = rows_to_tensors(rows[:split])
    X_va, y_va = rows_to_tensors(rows[split:])
    gate = KANGate(seed=seed)
    t0 = time.time()
    gate.fit(X_tr, y_tr, steps=KAN_STEPS, lamb=0.01, test_X=X_va, test_y=y_va)
    with torch.no_grad():
        spline_va = gate.model(X_va).numpy()[:, 0]
    formula = gate.formula()  # fixes every edge to its symbolic fit
    with torch.no_grad():
        symbolic_va = gate.model(X_va).numpy()[:, 0]
    info = {
        "train_seconds": time.time() - t0,
        "formula": formula,
        "val_mae_spline": float(np.abs(spline_va - y_va.numpy()[:, 0]).mean()),
        "val_mae_symbolic": float(np.abs(symbolic_va - y_va.numpy()[:, 0]).mean()),
    }
    return KANScorer(gate, name), info


def kan_interpretability(scorer: KANScorer, X: np.ndarray) -> dict:
    model = scorer.gate.model
    rhs = scorer.gate.formula().split("=", 1)[1].strip()
    expr = sympy.sympify(rhs)
    fn = sympy.lambdify([sympy.Symbol(n) for n in FEATURE_NAMES], expr, "numpy")
    with torch.no_grad():
        forward = model(torch.tensor(X, dtype=torch.float32)).numpy()[:, 0]
    by_formula = np.asarray(fn(*[X[:, k] for k in range(X.shape[1])]), dtype=np.float64)
    by_formula = np.broadcast_to(by_formula, forward.shape)
    active = sum(int((m.mask > 0).sum()) for m in model.symbolic_fun
                 ) - sum(1 for m in model.symbolic_fun for row in m.funs_name for f in row if f == "0")
    return {
        "faithfulness_max_abs": float(np.max(np.abs(forward - by_formula))),
        "formula_ops": int(sympy.count_ops(expr)),
        "formula_terms": len(sympy.Add.make_args(sympy.expand(expr))),
        "active_edges": int(active),
    }


# ----------------------------------------------------------------------------- outcomes

def outcome(run: dict, stop_epoch: int) -> dict:
    val, acc = run["val_loss"], run["val_acc"]
    best_full = min(val)
    seen = val[:stop_epoch]
    best = min(seen)
    best_idx = seen.index(best)
    regret = (best - best_full) / best_full
    return {
        "stop_epoch": stop_epoch,
        "regret": regret,
        "acc_drop_pp": 100 * (acc[val.index(best_full)] - acc[best_idx]),
        "premature": regret > REGRET_FAIL,
    }


def stop_by_scores(run: dict, scores) -> int:
    for p, s in zip(run["points"], scores):
        if s >= ADJUST_LR_MAX:
            return p["epochs_seen"]
    return len(run["val_loss"])


def stop_patience(run: dict, patience: int) -> int:
    val = run["val_loss"]
    for e in range(1, len(val) + 1):
        best_idx = val[:e].index(min(val[:e]))
        if e - (best_idx + 1) >= patience:
            return e
    return len(val)


def stop_median(run: dict, population: list[dict]) -> int:
    """Median stopping rule (Golovin et al., KDD 2017), applied on the same
    decision schedule as the gates, with the training-fold runs as the
    population of completed trials."""
    for p in run["points"]:
        t = p["epochs_seen"]
        median = statistics.median(sum(r["val_loss"][:t]) / t for r in population)
        if min(run["val_loss"][:t]) > median:
            return t
    return len(run["val_loss"])


def stop_oracle(run: dict) -> int:
    for p in run["points"]:
        if remaining_gain(run["val_loss"], p["epochs_seen"]) <= TAU * (1 - ADJUST_LR_MAX / 100):
            return p["epochs_seen"]
    return len(run["val_loss"])


def successive_halving(runs: list[dict], rungs=(4, 12, 36), eta: int = 3) -> dict:
    """Synchronous successive halving (the per-bracket core of Hyperband and
    ASHA) over one fold's configurations, replayed on recorded curves."""
    budget = len(runs[0]["val_loss"])
    alive = list(runs)
    trained = {r["run_id"]: 0 for r in runs}
    for rung in rungs:
        for r in alive:
            trained[r["run_id"]] = rung
        alive.sort(key=lambda r: min(r["val_loss"][:rung]))
        alive = alive[:max(1, len(alive) // eta)]
    for r in alive:
        trained[r["run_id"]] = budget
    best = min(min(r["val_loss"][:trained[r["run_id"]]]) for r in runs)
    return {"total_epochs": sum(trained.values()), "best_found": best, "per_run_epochs": trained}


# ----------------------------------------------------------------------------- summaries

def summarize(per_run: dict, budget: int) -> dict:
    vals = list(per_run.values())
    regrets = [v["regret"] for v in vals]
    return {
        "n": len(vals),
        "compute_saved_pct": 100 * (1 - sum(v["stop_epoch"] for v in vals) / (budget * len(vals))),
        "mean_regret_pct": 100 * statistics.mean(regrets),
        "median_regret_pct": 100 * statistics.median(regrets),
        "p90_regret_pct": 100 * float(np.percentile(regrets, 90)),
        "mean_acc_drop_pp": statistics.mean(v["acc_drop_pp"] for v in vals),
        "premature_rate_pct": 100 * statistics.mean(v["premature"] for v in vals),
    }


def mean_std(dicts: list[dict]) -> dict:
    keys = dicts[0].keys()
    return {k: {"mean": statistics.mean(d[k] for d in dicts),
                "std": statistics.pstdev(d[k] for d in dicts) if len(dicts) > 1 else 0.0}
            for k in keys}


def fidelity(scorer: Scorer, test_runs: list[dict]) -> dict:
    X = np.concatenate([r["X"] for r in test_runs])
    y = np.array([p["target"] for r in test_runs for p in r["points"]])
    pred = scorer.predict(X)
    ss_res = float(((pred - y) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    yb, pb = banded(y), banded(pred)
    f1s = []
    for c in (0, 1, 2):
        tp = int(((pb == c) & (yb == c)).sum())
        fp = int(((pb == c) & (yb != c)).sum())
        fn = int(((pb != c) & (yb == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"mae": float(np.abs(pred - y).mean()), "r2": 1 - ss_res / ss_tot,
            "band_accuracy": float((yb == pb).mean()), "band_macro_f1": float(np.mean(f1s))}


def latency_us(scorer: Scorer, x: np.ndarray, n: int = 300) -> float:
    one = x[:1]
    scorer.predict(one)
    t0 = time.perf_counter()
    for _ in range(n):
        scorer.predict(one)
    return 1e6 * (time.perf_counter() - t0) / n


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=CURVES)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "results.json")
    parser.add_argument("--quick", action="store_true", help="Smoke test: 1 gate seed, short KAN fits.")
    args = parser.parse_args()
    global GATE_SEEDS, KAN_STEPS
    if args.quick:
        GATE_SEEDS, KAN_STEPS = [0], 20

    runs = load_runs(args.curves)
    budget = len(runs[0]["val_loss"])
    seeds = sorted({r["seed"] for r in runs})
    print(f"{len(runs)} runs, budget {budget} epochs, folds by seed {seeds}", flush=True)

    # Teacher-trained KAN (the original deployed gate): synthetic curves +
    # reference-policy labels, independent of the real folds.
    synth_rows = build_synthetic_dataset()
    teacher_kans = [fit_kan(synth_rows, s, "KAN (teacher labels)") for s in GATE_SEEDS]

    per_policy: dict[str, list[dict]] = {}   # name -> per gate-seed dict run_id -> outcome
    fold_scorers: dict[int, dict[str, list[Scorer]]] = {}
    fid: dict[str, list[dict]] = {}
    kan_info: list[dict] = []
    interp: dict[str, list[dict]] = {}
    portfolio: dict[str, list[dict]] = {}
    examples = []

    def record(name: str, seed_idx: int, run: dict, stop: int):
        bucket = per_policy.setdefault(name, [dict() for _ in GATE_SEEDS])
        bucket[seed_idx][run["run_id"]] = outcome(run, stop)

    for test_seed in seeds:
        train_runs = [r for r in runs if r["seed"] != test_seed]
        test_runs = [r for r in runs if r["seed"] == test_seed]
        rows = rows_from(train_runs, TAU)
        X_tr, y_tr = rows_to_tensors(rows)
        X_tr, y_tr = X_tr.numpy(), y_tr.numpy()[:, 0]
        print(f"fold test_seed={test_seed}: {len(rows)} training decision points", flush=True)

        scorers: dict[str, list[Scorer]] = {}
        for si, s in enumerate(GATE_SEEDS):
            kan, info = fit_kan(rows, s, "KAN (hindsight labels, ours)")
            kan_info.append({"test_seed": test_seed, "gate_seed": s, **info})
            interp.setdefault("KAN (hindsight labels, ours)", []).append(
                kan_interpretability(kan, np.concatenate([r["X"] for r in test_runs])))
            if test_seed == seeds[-1] and s == GATE_SEEDS[0]:
                kan.gate.save(RESULTS_DIR / "gates" / f"kan_hindsight_test_seed{test_seed}")
            mlp = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=3000, random_state=s,
                               early_stopping=True).fit(X_tr, y_tr)
            scorers.setdefault("KAN (hindsight labels, ours)", []).append(kan)
            scorers.setdefault("MLP (hindsight labels)", []).append(
                SkScorer(mlp, "MLP (hindsight labels)", "mlp"))
            scorers.setdefault("KAN (teacher labels)", []).append(teacher_kans[si][0])
        ridge = Ridge(alpha=1e-3).fit(X_tr, y_tr)
        tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_tr, y_tr)
        for name, model, kind in (("Linear (hindsight labels)", ridge, "linear"),
                                  ("Tree depth-3 (hindsight labels)", tree, "tree")):
            scorers[name] = [SkScorer(model, name, kind)] * len(GATE_SEEDS)
        scorers["Reference rule (teacher)"] = [RuleScorer()] * len(GATE_SEEDS)
        fold_scorers[test_seed] = scorers

        for name, lst in scorers.items():
            for si, sc in enumerate(lst):
                fid.setdefault(name, []).append(fidelity(sc, test_runs))
                for run in test_runs:
                    record(name, si, run, stop_by_scores(run, sc.predict(run["X"])))
        for run in test_runs:
            for si in range(len(GATE_SEEDS)):
                record("No early stopping", si, run, budget)
                record("Hindsight oracle", si, run, stop_oracle(run))
                for p in (3, 5, 8):
                    record(f"Patience-{p}", si, run, stop_patience(run, p))
                record("Median stopping rule", si, run, stop_median(run, train_runs))

        # Portfolio view: the fold's 12 configurations as one tuning job.
        best_possible = min(min(r["val_loss"]) for r in test_runs)
        sh = successive_halving(test_runs)
        portfolio.setdefault("Successive halving (eta=3)", []).append(
            {"test_seed": test_seed, "total_epochs": sh["total_epochs"],
             "best_found": sh["best_found"], "best_possible": best_possible})

        # Worked examples for the paper's trajectory figure.
        kan0 = scorers["KAN (hindsight labels, ours)"][0]
        for run in test_runs:
            examples.append({
                "run_id": run["run_id"], "lr": run["lr"], "batch_size": run["batch_size"],
                "seed": run["seed"], "val_loss": run["val_loss"], "val_acc": run["val_acc"],
                "check_epochs": [p["epochs_seen"] for p in run["points"]],
                "kan_scores": kan0.predict(run["X"]).tolist(),
                "hindsight_targets": [p["target"] for p in run["points"]],
                "stops": {
                    "KAN (hindsight labels, ours)": stop_by_scores(run, kan0.predict(run["X"])),
                    "Patience-5": stop_patience(run, 5),
                    "Median stopping rule": stop_median(run, train_runs),
                    "Hindsight oracle": stop_oracle(run),
                },
            })

    # Per-run policies also give a portfolio result: run every config under the policy.
    for name, seeds_buckets in per_policy.items():
        for test_seed in seeds:
            test_ids = [r["run_id"] for r in runs if r["seed"] == test_seed]
            res = seeds_buckets[0]
            total = sum(res[i]["stop_epoch"] for i in test_ids)
            best_found = min(min(next(r for r in runs if r["run_id"] == i)["val_loss"][:res[i]["stop_epoch"]])
                             for i in test_ids)
            best_possible = min(min(r["val_loss"]) for r in runs if r["seed"] == test_seed)
            portfolio.setdefault(name, []).append({"test_seed": test_seed, "total_epochs": total,
                                                   "best_found": best_found,
                                                   "best_possible": best_possible})

    policies = []
    for name, buckets in per_policy.items():
        summaries = [summarize(b, budget) for b in buckets]
        policies.append({"name": name, "summary": mean_std(summaries), "per_run": buckets[0]})

    portfolio_summary = []
    for name, folds in portfolio.items():
        total = sum(f["total_epochs"] for f in folds)
        portfolio_summary.append({
            "name": name, "folds": folds,
            "compute_saved_pct": 100 * (1 - total / (budget * len(runs))),
            "mean_best_regret_pct": 100 * statistics.mean(
                (f["best_found"] - f["best_possible"]) / f["best_possible"] for f in folds),
        })

    # Paired tests on per-run regret and stop epoch (gate seed 0).
    def per_run_values(name, key):
        res = per_policy[name][0]
        return [res[r["run_id"]][key] for r in runs]

    ours = "KAN (hindsight labels, ours)"
    stats = {}
    for other in ("Patience-5", "Patience-3", "Median stopping rule", "KAN (teacher labels)",
                  "MLP (hindsight labels)", "Linear (hindsight labels)", "Reference rule (teacher)"):
        entry = {}
        for key in ("regret", "stop_epoch"):
            a, b = per_run_values(ours, key), per_run_values(other, key)
            diffs = [x - y for x, y in zip(a, b)]
            if all(abs(d) < 1e-12 for d in diffs):
                entry[key] = {"p_value": 1.0, "median_diff": 0.0}
            else:
                entry[key] = {"p_value": float(wilcoxon(a, b, zero_method="zsplit").pvalue),
                              "median_diff": float(statistics.median(diffs))}
        stats[other] = entry

    # Sensitivity of the trade-off to the hindsight tolerance.
    tau_sweep = []
    for tau in ((0.05,) if args.quick else (0.02, 0.05, 0.10)):
        kan_buckets, lin_buckets = {}, {}
        for test_seed in seeds:
            train_runs = [r for r in runs if r["seed"] != test_seed]
            test_runs = [r for r in runs if r["seed"] == test_seed]
            rows = rows_from(train_runs, tau)
            kan, _ = fit_kan(rows, 0, "kan")
            X_tr, y_tr = rows_to_tensors(rows)
            lin = SkScorer(Ridge(alpha=1e-3).fit(X_tr.numpy(), y_tr.numpy()[:, 0]), "lin", "linear")
            for run in test_runs:
                kan_buckets[run["run_id"]] = outcome(run, stop_by_scores(run, kan.predict(run["X"])))
                lin_buckets[run["run_id"]] = outcome(run, stop_by_scores(run, lin.predict(run["X"])))
        tau_sweep.append({"tau": tau, "kan": summarize(kan_buckets, budget),
                          "linear": summarize(lin_buckets, budget)})
        print(f"tau={tau}: kan {tau_sweep[-1]['kan']}", flush=True)

    # Interpretability + latency summaries.
    any_X = np.concatenate([r["X"] for r in runs])
    some = fold_scorers[seeds[0]]
    teacher_interp = [kan_interpretability(k, any_X) for k, _ in teacher_kans]
    interpretability = {
        "KAN (hindsight labels, ours)": mean_std(interp["KAN (hindsight labels, ours)"]),
        "KAN (teacher labels)": mean_std(teacher_interp),
        "Linear (hindsight labels)": {"formula_terms": len(FEATURE_NAMES) + 1},
        "Tree depth-3 (hindsight labels)": {
            "leaves": int(some["Tree depth-3 (hindsight labels)"][0].model.get_n_leaves())},
        "MLP (hindsight labels)": {"parameters": int(sum(
            w.size for w in some["MLP (hindsight labels)"][0].model.coefs_) + sum(
            b.size for b in some["MLP (hindsight labels)"][0].model.intercepts_))},
    }
    latency = {name: latency_us(lst[0], any_X) for name, lst in some.items()}

    fidelity_summary = {name: mean_std(v) for name, v in fid.items()}

    out = {
        "config": {"tau": TAU, "regret_fail": REGRET_FAIL, "check_every": DEFAULT_CHECK_EVERY,
                   "first_check": DEFAULT_FIRST_CHECK, "budget_epochs": budget,
                   "gate_seeds": GATE_SEEDS, "kan_steps": KAN_STEPS,
                   "stop_threshold": ADJUST_LR_MAX, "n_runs": len(runs),
                   "lrs": sorted({r["lr"] for r in runs}),
                   "batch_sizes": sorted({r["batch_size"] for r in runs}), "seeds": seeds,
                   "subset_size": runs[0]["subset_size"], "val_size": runs[0]["val_size"]},
        "policies": policies,
        "portfolio": portfolio_summary,
        "fidelity": fidelity_summary,
        "kan_training": kan_info,
        "teacher_kan_formulas": [info["formula"] for _, info in teacher_kans],
        "interpretability": interpretability,
        "latency_us": latency,
        "stats": stats,
        "tau_sweep": tau_sweep,
        "examples": examples,
        "curves_summary": [{"run_id": r["run_id"], "lr": r["lr"], "batch_size": r["batch_size"],
                            "seed": r["seed"], "best_val_loss": min(r["val_loss"]),
                            "best_epoch": r["val_loss"].index(min(r["val_loss"])) + 1,
                            "best_val_acc": max(r["val_acc"]),
                            "mean_epoch_seconds": statistics.mean(r["epoch_seconds"])}
                           for r in runs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out}")
    for p in sorted(policies, key=lambda p: -p["summary"]["compute_saved_pct"]["mean"]):
        s = p["summary"]
        print(f"{p['name']:<34} saved {s['compute_saved_pct']['mean']:5.1f}%  "
              f"regret {s['mean_regret_pct']['mean']:5.2f}%  premature {s['premature_rate_pct']['mean']:5.1f}%")


if __name__ == "__main__":
    main()
