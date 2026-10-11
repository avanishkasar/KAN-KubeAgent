"""Generalisation benchmark: four real workloads, two evaluation protocols.

experiments/benchmark.py evaluates stopping policies on one workload
(Fashion-MNIST, small CNN, Adam). This script asks whether the result
holds on other real training runs, and whether a gate trained on some
workloads can control a workload it has never seen.

Workloads (experiments/collect_curves.py, 36 uncontrolled runs each:
6 learning rates x 2 batch sizes x 3 seeds, 40-epoch budget):
    fmnist_cnn       Fashion-MNIST, two-conv CNN, Adam
    mnist_cnn        MNIST, two-conv CNN, Adam
    cifar10_cnn      CIFAR-10, three-conv CNN, Adam
    fmnist_mlp_sgd   Fashion-MNIST, 256-128 MLP, SGD with momentum

Protocols:
    within  3-fold cross-validation inside each workload, folds grouped by
            training seed (train on two seeds, test on the third).
    lowo    leave-one-workload-out: train on all runs of three workloads,
            test on every run of the fourth.

Every policy is replayed on the recorded curves (counterfactual replay,
as in experiments/benchmark.py). Reported per policy: compute saved,
regret in best validation loss, premature-stop rate, accuracy drop, with
run-level bootstrap 95% confidence intervals and paired Wilcoxon tests
against the KAN gate (Holm-corrected). The rule-label baseline is the
same KAN trained on the same real training-fold decision points, labelled
by the hand-written reference policy instead of in hindsight, so the two
differ only in the label source. Also: stop-threshold and patience
sweeps (trade-off frontiers), a feature ablation, a tolerance sweep, the
faithfulness of every extracted formula, and the gates used by the
closed-loop experiment on unseen workloads.

    python -m experiments.generalization
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from experiments import benchmark as B
from experiments.collect_curves import WORKLOADS, curves_path
from kan_gate.features import FEATURE_NAMES
from kan_gate.gate import ADJUST_LR_MAX
from kan_gate.reference_policy import reference_decision
from kan_gate.train import rows_to_tensors

RESULTS_DIR = B.RESULTS_DIR
OUT = RESULTS_DIR / "generalization.json"

WORKLOAD_LABELS = {
    "fmnist_cnn": "Fashion-MNIST / CNN / Adam",
    "mnist_cnn": "MNIST / CNN / Adam",
    "cifar10_cnn": "CIFAR-10 / CNN / Adam",
    "fmnist_mlp_sgd": "Fashion-MNIST / MLP / SGD",
}
OURS = "KAN gate (hindsight labels)"
GATE_SEEDS = [0, 1, 2]
N_BOOT = 10000
THRESHOLDS = [55, 60, 65, 70, 75, 80, 85, 90, 95]
PATIENCES = list(range(1, 21))


# ----------------------------------------------------------------------------- data

def load_workloads(names=None) -> dict[str, list[dict]]:
    out = {}
    for w in names or WORKLOADS:
        runs = B.load_runs(curves_path(w))
        if len(runs) != 36:
            raise SystemExit(f"{w}: expected 36 runs, found {len(runs)} - run "
                             f"python -m experiments.collect_curves --workload {w}")
        for r in runs:
            r["workload"] = w
            r["run_id"] = f"{w}/{r['run_id']}"
        out[w] = runs
    return out


def masked(runs: list[dict], drop: str | None, means: np.ndarray | None) -> list[np.ndarray]:
    """Feature matrices of runs, with one feature replaced by its training
    mean (feature ablation) when `drop` is set."""
    mats = []
    for r in runs:
        X = r["X"].copy()
        if drop is not None:
            X[:, FEATURE_NAMES.index(drop)] = means[FEATURE_NAMES.index(drop)]
        mats.append(X)
    return mats


def stop_by_threshold(run: dict, scores, threshold: float) -> int:
    for p, s in zip(run["points"], scores):
        if s >= threshold:
            return p["epochs_seen"]
    return len(run["val_loss"])


# ----------------------------------------------------------------------------- one split

def evaluate_split(train_runs, test_runs, population_for, tau=B.TAU,
                   drop=None, gate_seeds=None, full=True):
    """Fit every learned policy on train_runs and replay every policy on
    test_runs. Returns {policy: [ {run_id: outcome} per gate seed ]} plus
    per-run score traces for the threshold sweep and the fitted KAN infos."""
    gate_seeds = gate_seeds or GATE_SEEDS
    rows = B.rows_from(train_runs, tau)
    X_tr, y_tr = rows_to_tensors(rows)
    X_np, y_np = X_tr.numpy(), y_tr.numpy()[:, 0]
    means = X_np.mean(axis=0)
    if drop is not None:
        X_np = X_np.copy()
        X_np[:, FEATURE_NAMES.index(drop)] = means[FEATURE_NAMES.index(drop)]
        rows = [({**f, drop: float(means[FEATURE_NAMES.index(drop)])}, t) for f, t in rows]
    test_X = masked(test_runs, drop, means)

    res: dict[str, list[dict]] = {}
    traces: dict[str, list[dict]] = {}
    infos = []

    def put(name, si, run, stop):
        bucket = res.setdefault(name, [dict() for _ in gate_seeds])
        bucket[si][run["run_id"]] = B.outcome(run, stop)

    def score_all(name, si, scorer):
        for run, X in zip(test_runs, test_X):
            s = scorer.predict(X)
            traces.setdefault(name, [dict() for _ in gate_seeds])[si][run["run_id"]] = s
            put(name, si, run, B.stop_by_scores(run, s))

    for si, seed in enumerate(gate_seeds):
        kan, info = B.fit_kan(rows, seed, OURS)
        info["interp"] = B.kan_interpretability(kan, np.concatenate(test_X))
        infos.append({"kan": kan, **info})
        score_all(OURS, si, kan)
        if not full:
            continue
        mlp = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=3000, random_state=seed,
                           early_stopping=True).fit(X_np, y_np)
        score_all("MLP (hindsight labels)", si, B.SkScorer(mlp, "mlp", "mlp"))
        rule_rows = [(f, reference_decision(f)) for f, _ in rows]
        score_all("KAN gate (rule labels)", si, B.fit_kan(rule_rows, seed, "rule")[0])
    if not full:
        return res, traces, infos

    ridge = Ridge(alpha=1e-3).fit(X_np, y_np)
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_np, y_np)
    for si in range(len(gate_seeds)):
        score_all("Linear (hindsight labels)", si, B.SkScorer(ridge, "lin", "linear"))
        score_all("Tree depth-3 (hindsight labels)", si, B.SkScorer(tree, "tree", "tree"))
        score_all("Reference rule", si, B.RuleScorer())
        for run in test_runs:
            put("No early stopping", si, run, len(run["val_loss"]))
            put("Hindsight oracle", si, run, B.stop_oracle(run))
            for p in (3, 5, 8):
                put(f"Patience-{p}", si, run, B.stop_patience(run, p))
            put("Median stopping rule", si, run, B.stop_median(run, population_for(run)))
    return res, traces, infos


def merge(into: dict, part: dict):
    for name, buckets in part.items():
        dst = into.setdefault(name, [dict() for _ in buckets])
        for d, s in zip(dst, buckets):
            d.update(s)


# ----------------------------------------------------------------------------- statistics

def per_run_mean(buckets: list[dict], key: str) -> dict[str, float]:
    ids = buckets[0].keys()
    return {i: statistics.mean(float(b[i][key]) for b in buckets) for i in ids}


def bootstrap_ci(values: np.ndarray, stat, n=N_BOOT, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    stats = np.array([stat(values[i]) for i in idx])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarize_policy(buckets: list[dict], budget: int, run_ids: list[str]) -> dict:
    stop = per_run_mean(buckets, "stop_epoch")
    regret = per_run_mean(buckets, "regret")
    prem = per_run_mean(buckets, "premature")
    acc = per_run_mean(buckets, "acc_drop_pp")
    s = np.array([stop[i] for i in run_ids])
    g = np.array([regret[i] for i in run_ids])
    saved = lambda v: 100 * (1 - v.mean() / budget)  # noqa: E731
    per_seed = [B.summarize({i: b[i] for i in run_ids}, budget) for b in buckets]
    return {
        "n_runs": len(run_ids),
        "compute_saved_pct": saved(s),
        "compute_saved_ci": bootstrap_ci(s, saved),
        "compute_saved_seed_std": statistics.pstdev(p["compute_saved_pct"] for p in per_seed),
        "mean_regret_pct": 100 * g.mean(),
        "mean_regret_ci": [100 * x for x in bootstrap_ci(g, np.mean, seed=1)],
        "p90_regret_pct": 100 * float(np.percentile(g, 90)),
        "max_regret_pct": 100 * float(g.max()),
        "premature_rate_pct": 100 * statistics.mean(prem[i] for i in run_ids),
        "mean_acc_drop_pp": statistics.mean(acc[i] for i in run_ids),
    }


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    m, running, out = len(order), 0.0, {}
    for k, name in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvals[name]))
        out[name] = running
    return out


def paired_tests(results: dict, run_ids: list[str], others: list[str]) -> dict:
    out = {}
    for key in ("regret", "stop_epoch"):
        ours = per_run_mean(results[OURS], key)
        raw, med = {}, {}
        for other in others:
            theirs = per_run_mean(results[other], key)
            a = np.array([ours[i] for i in run_ids])
            b = np.array([theirs[i] for i in run_ids])
            d = a - b
            med[other] = float(np.median(d)) * (100 if key == "regret" else 1)
            raw[other] = 1.0 if np.all(np.abs(d) < 1e-12) else float(
                wilcoxon(a, b, zero_method="zsplit").pvalue)
        adj = holm(raw)
        out[key] = {o: {"p_raw": raw[o], "p_holm": adj[o], "median_diff": med[o]} for o in others}
    return out


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="1 gate seed, short fits, no sweeps.")
    parser.add_argument("--workloads", nargs="+", choices=sorted(WORKLOADS), default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    global GATE_SEEDS, N_BOOT
    if args.quick:
        GATE_SEEDS, N_BOOT, B.KAN_STEPS = [0], 200, 20

    t_start = time.time()
    data = load_workloads(args.workloads)
    all_runs = [r for w in data for r in data[w]]
    budget = len(all_runs[0]["val_loss"])
    run_ids = [r["run_id"] for r in all_runs]
    by_id = {r["run_id"]: r for r in all_runs}
    print(f"{len(all_runs)} runs in {len(data)} workloads, budget {budget}", flush=True)

    def population_for(run):
        """Median rule's population: completed runs of the same workload
        with other seeds (the trials a tuning job would already have)."""
        return [r for r in data[run["workload"]] if r["seed"] != run["seed"]]

    protocols = {}
    kan_infos = {"within": [], "lowo": []}
    traces = {"within": {}, "lowo": {}}

    # within-workload, seed-grouped 3-fold CV
    res_w: dict = {}
    for w, runs in data.items():
        for test_seed in sorted({r["seed"] for r in runs}):
            tr = [r for r in runs if r["seed"] != test_seed]
            te = [r for r in runs if r["seed"] == test_seed]
            part, tr_scores, infos = evaluate_split(tr, te, population_for)
            merge(res_w, part)
            merge(traces["within"], tr_scores)
            kan_infos["within"] += [{"workload": w, "test_seed": test_seed,
                                     **{k: v for k, v in i.items() if k != "kan"}} for i in infos]
        print(f"within {w} done ({time.time() - t_start:.0f}s)", flush=True)
    protocols["within"] = res_w

    # leave-one-workload-out
    res_l: dict = {}
    for w in data:
        tr = [r for v in data if v != w for r in data[v]]
        part, tr_scores, infos = evaluate_split(tr, data[w], population_for)
        merge(res_l, part)
        merge(traces["lowo"], tr_scores)
        if not args.quick:
            infos[0]["kan"].gate.save(RESULTS_DIR / "gates" / f"kan_lowo_{w}")
        kan_infos["lowo"] += [{"workload": w, **{k: v for k, v in i.items() if k != "kan"}}
                              for i in infos]
        print(f"lowo held-out {w} done ({time.time() - t_start:.0f}s)", flush=True)
    protocols["lowo"] = res_l

    # summaries
    summary = {}
    per_workload = {}
    tests = {}
    others = ["Patience-3", "Patience-5", "Patience-8", "Median stopping rule",
              "KAN gate (rule labels)", "Reference rule", "Linear (hindsight labels)",
              "Tree depth-3 (hindsight labels)", "MLP (hindsight labels)"]
    for proto, res in protocols.items():
        summary[proto] = {name: summarize_policy(b, budget, run_ids) for name, b in res.items()}
        per_workload[proto] = {
            w: {name: summarize_policy(b, budget, [r["run_id"] for r in data[w]])
                for name, b in res.items()} for w in data}
        tests[proto] = paired_tests(res, run_ids, others)

    # trade-off frontiers (stop-threshold sweep for learned gates, patience sweep)
    frontiers = {}
    for proto in protocols:
        fr = {}
        for name in (OURS, "Linear (hindsight labels)", "MLP (hindsight labels)"):
            pts = []
            for th in THRESHOLDS:
                buckets = []
                for seed_traces in traces[proto][name]:
                    buckets.append({i: B.outcome(by_id[i], stop_by_threshold(by_id[i], s, th))
                                    for i, s in seed_traces.items()})
                sm = summarize_policy(buckets, budget, run_ids)
                pts.append({"threshold": th, "saved": sm["compute_saved_pct"],
                            "regret": sm["mean_regret_pct"], "premature": sm["premature_rate_pct"]})
            fr[name] = pts
        frontiers[proto] = fr
    pat = []
    for p in PATIENCES:
        b = [{r["run_id"]: B.outcome(r, B.stop_patience(r, p)) for r in all_runs}]
        sm = summarize_policy(b, budget, run_ids)
        pat.append({"patience": p, "saved": sm["compute_saved_pct"], "regret": sm["mean_regret_pct"],
                    "premature": sm["premature_rate_pct"]})
    frontiers["patience"] = pat

    # feature ablation and tolerance sweep (lowo, KAN only)
    ablation, tau_sweep = [], []
    if not args.quick:
        for drop in [None] + FEATURE_NAMES:
            res: dict = {}
            for w in data:
                tr = [r for v in data if v != w for r in data[v]]
                part, _, _ = evaluate_split(tr, data[w], population_for, drop=drop, full=False)
                merge(res, part)
            sm = summarize_policy(res[OURS], budget, run_ids)
            ablation.append({"removed_feature": drop, **sm})
            print(f"ablation {drop}: saved {sm['compute_saved_pct']:.1f} regret "
                  f"{sm['mean_regret_pct']:.2f}", flush=True)
        for tau in (0.02, 0.05, 0.10):
            res = {}
            for w in data:
                tr = [r for v in data if v != w for r in data[v]]
                part, _, _ = evaluate_split(tr, data[w], population_for, tau=tau, full=False)
                merge(res, part)
            sm = summarize_policy(res[OURS], budget, run_ids)
            tau_sweep.append({"tau": tau, **sm})
            print(f"tau {tau}: saved {sm['compute_saved_pct']:.1f} regret "
                  f"{sm['mean_regret_pct']:.2f}", flush=True)

    # interpretability of every fitted gate
    interp_all = [i["interp"] for p in kan_infos.values() for i in p]
    interpretability = {
        "n_gates": len(interp_all),
        "faithfulness_max_abs": max(i["faithfulness_max_abs"] for i in interp_all),
        "faithfulness_median": statistics.median(i["faithfulness_max_abs"] for i in interp_all),
        "formula_terms_median": statistics.median(i["formula_terms"] for i in interp_all),
        "formula_terms_max": max(i["formula_terms"] for i in interp_all),
        "active_edges_median": statistics.median(i["active_edges"] for i in interp_all),
        "val_mae_spline_mean": statistics.mean(i["val_mae_spline"] for p in kan_infos.values() for i in p),
        "val_mae_symbolic_mean": statistics.mean(i["val_mae_symbolic"] for p in kan_infos.values() for i in p),
        "train_seconds_mean": statistics.mean(i["train_seconds"] for p in kan_infos.values() for i in p),
        "lowo_formulas": {i["workload"]: i["formula"] for i in kan_infos["lowo"][::len(GATE_SEEDS)]},
    }

    workload_stats = {}
    for w, runs in data.items():
        best = [min(r["val_loss"]) for r in runs]
        best_ep = [r["val_loss"].index(min(r["val_loss"])) + 1 for r in runs]
        workload_stats[w] = {
            "label": WORKLOAD_LABELS[w], "n_runs": len(runs),
            "lrs": sorted({r["lr"] for r in runs}),
            "best_val_acc_max": max(max(r["val_acc"]) for r in runs),
            "best_val_acc_median": statistics.median(max(r["val_acc"]) for r in runs),
            "best_epoch_median": statistics.median(best_ep),
            "best_epoch_min": min(best_ep), "best_epoch_max": max(best_ep),
            "frac_best_in_last_5": statistics.mean(e > budget - 5 for e in best_ep),
            "best_val_loss_median": statistics.median(best),
            "epoch_seconds_mean": statistics.mean(statistics.mean(r["epoch_seconds"]) for r in runs),
            "decision_points": sum(len(r["points"]) for r in runs),
        }

    keep = [OURS, "Linear (hindsight labels)", "MLP (hindsight labels)", "Patience-5",
            "Patience-8", "Hindsight oracle", "KAN gate (rule labels)"]
    per_run = {proto: {name: {i: {"stop_epoch": [b[i]["stop_epoch"] for b in res[name]],
                                  "regret": [b[i]["regret"] for b in res[name]]}
                              for i in run_ids}
                       for name in keep}
               for proto, res in protocols.items()}
    score_traces = {proto: {i: [float(v) for v in traces[proto][OURS][0][i]] for i in run_ids}
                    for proto in protocols}
    curves = {i: {"val_loss": by_id[i]["val_loss"], "val_acc": by_id[i]["val_acc"],
                  "lr": by_id[i]["lr"], "batch_size": by_id[i]["batch_size"],
                  "seed": by_id[i]["seed"], "workload": by_id[i]["workload"],
                  "check_epochs": [p["epochs_seen"] for p in by_id[i]["points"]],
                  "targets": [p["target"] for p in by_id[i]["points"]]} for i in run_ids}

    out = {
        "per_run": per_run,
        "score_traces": score_traces,
        "curves": curves,
        "config": {"workloads": list(data), "budget_epochs": budget, "tau": B.TAU,
                   "regret_fail": B.REGRET_FAIL, "stop_threshold": ADJUST_LR_MAX,
                   "gate_seeds": GATE_SEEDS, "kan_steps": B.KAN_STEPS, "n_boot": N_BOOT,
                   "n_runs": len(all_runs)},
        "workloads": workload_stats,
        "summary": summary,
        "per_workload": per_workload,
        "tests": tests,
        "frontiers": frontiers,
        "ablation": ablation,
        "tau_sweep": tau_sweep,
        "interpretability": interpretability,
        "seconds": time.time() - t_start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out} in {time.time() - t_start:.0f}s")
    for proto in protocols:
        print(f"--- {proto}")
        for name, s in sorted(summary[proto].items(), key=lambda kv: -kv[1]["compute_saved_pct"]):
            print(f"{name:<34} saved {s['compute_saved_pct']:5.1f}%  regret {s['mean_regret_pct']:6.2f}%  "
                  f"premature {s['premature_rate_pct']:5.1f}%")


if __name__ == "__main__":
    main()
