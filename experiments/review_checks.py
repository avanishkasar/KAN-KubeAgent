"""Additional offline checks, all under the leave-one-workload-out protocol
with the same three gate seeds as experiments/generalization.py:

1. Decision cadence. The gate decides every two epochs and patience after
   every epoch in the main study. Here both are also run at the other
   cadence: the gate at every epoch from epoch 4, patience only at the
   gate's decision epochs (4, 6, ..., 38).
2. Learning-curve extrapolation baseline. At each decision epoch a
   power law v(t) = a + b * t^(-c) is fitted to the validation losses seen
   so far; the run stops when the predicted improvement over the rest of
   the budget is below tau/4 = 1.25%, the same quantity the gate's stop band
   targets. No training, no tuning.
3. Decision-level cost of symbolic fixing. The spline network (before its
   edges are fixed) and the symbolic gate are applied to the same held-out
   decision points; we report how often their band decisions differ and
   the stopping outcome of each.
4. Sensitivity to the feature constants: window w in {3, 5, 8} and the
   epochs-since-improvement normaliser in {20, 30, 40}.

Writes experiments/results/review_checks.json.

    python -m experiments.review_checks
"""
from __future__ import annotations

import functools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon

import kan_gate.features as F
import kan_gate.hindsight as H
from experiments import benchmark as B
from experiments.collect_curves import WORKLOADS
from experiments.generalization import GATE_SEEDS, load_workloads
from kan_gate.features import FEATURE_NAMES

OUT = Path(__file__).resolve().parents[1] / "experiments" / "results" / "review_checks.json"
T_FIRST, T_EVERY = H.DEFAULT_FIRST_CHECK, H.DEFAULT_CHECK_EVERY


def features_at(run, every: int):
    pts = H.decision_points(run["val_loss"], run["grad_norm"], every, T_FIRST, B.TAU)
    return pts, np.array([[p["features"][n] for n in FEATURE_NAMES] for p in pts], dtype=np.float32)


def stop_from_scores(points, scores, total):
    for p, s in zip(points, scores):
        if s >= B.ADJUST_LR_MAX:
            return p["epochs_seen"]
    return total


def patience_at_decisions(run, patience: int) -> int:
    val = run["val_loss"]
    for e in range(T_FIRST, len(val), T_EVERY):
        best_idx = val[:e].index(min(val[:e]))
        if e - (best_idx + 1) >= patience:
            return e
    return len(val)


def _pow3(t, a, b, c):
    return a + b * np.power(t, -c)


def stop_extrapolation(run) -> int:
    """Stop when the fitted power law predicts < tau/4 relative improvement
    of the best validation loss over the remaining budget."""
    val = np.asarray(run["val_loss"], dtype=float)
    T = len(val)
    for e in range(T_FIRST, T, T_EVERY):
        t = np.arange(1, e + 1, dtype=float)
        y = val[:e]
        best = y.min()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = curve_fit(_pow3, t, y, p0=(y[-1], max(y[0] - y[-1], 1e-3), 0.5),
                                 bounds=([0.0, -np.inf, 0.0], [np.inf, np.inf, 5.0]), maxfev=5000)
            future = _pow3(np.arange(e + 1, T + 1, dtype=float), *p)
            predicted_gain = max(0.0, best - future.min()) / best
        except (RuntimeError, ValueError):
            predicted_gain = 1.0  # no fit: keep training
        if predicted_gain < B.TAU * (1 - B.ADJUST_LR_MAX / 100):
            return e
    return T


def summarise(outcomes: dict[str, list[dict]], T: int) -> dict:
    """outcomes: run_id -> list of outcome dicts (one per gate seed)."""
    saved = [np.mean([1 - o["stop_epoch"] / T for o in v]) for v in outcomes.values()]
    regret = [np.mean([o["regret"] for o in v]) for v in outcomes.values()]
    prem = [np.mean([o["premature"] for o in v]) for v in outcomes.values()]
    return {"saved": 100 * float(np.mean(saved)), "regret": 100 * float(np.mean(regret)),
            "premature": 100 * float(np.mean(prem)), "n_runs": len(outcomes)}


def per_run(outcomes, key):
    return {k: float(np.mean([o[key] for o in v])) for k, v in outcomes.items()}


def run_lowo(W, label, collect_extra: bool):
    """LOWO with the current feature settings. Returns outcome dicts per policy."""
    res = {k: {} for k in ["kan", "kan_every_epoch", "kan_spline"]}
    agree, n_pts = 0, 0
    T = len(next(iter(W.values()))[0]["val_loss"])
    for held in WORKLOADS:
        train = [r for w in WORKLOADS if w != held for r in W[w]]
        rows = B.rows_from(train, B.TAU)
        test = W[held]
        cache2 = [features_at(r, T_EVERY) for r in test]
        cache1 = [features_at(r, 1) for r in test] if collect_extra else None
        for seed in GATE_SEEDS:
            # replicate benchmark.fit_kan, keeping a copy of the spline network
            g = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(rows), generator=g).tolist()
            rr = [rows[i] for i in order]
            split = int(len(rr) * 0.8)
            X_tr, y_tr = B.rows_to_tensors(rr[:split])
            X_va, y_va = B.rows_to_tensors(rr[split:])
            gate = B.KANGate(seed=seed)
            gate.fit(X_tr, y_tr, steps=B.KAN_STEPS, lamb=0.01, test_X=X_va, test_y=y_va)
            spline_scores = []
            if collect_extra:  # spline network's scores, before its edges are fixed
                for pts, X in cache2:
                    with torch.no_grad():
                        spline_scores.append(np.clip(gate.model(torch.tensor(X)).numpy()[:, 0], 0, 100))
            # benchmark.fit_kan evaluates the network on its validation split just before
            # fixing the edges, and pykan fits the symbolic functions to the activations of
            # the most recent forward pass, so repeat that pass to reproduce the same gate
            with torch.no_grad():
                gate.model(X_va)
            gate.formula()
            for k, (r, (pts, X)) in enumerate(zip(test, cache2)):
                with torch.no_grad():
                    s = np.clip(gate.model(torch.tensor(X)).numpy()[:, 0], 0, 100)
                res["kan"].setdefault(r["run_id"], []).append(B.outcome(r, stop_from_scores(pts, s, T)))
                if collect_extra:
                    sp = spline_scores[k]
                    agree += int((B.banded(sp) == B.banded(s)).sum())
                    n_pts += len(s)
                    res["kan_spline"].setdefault(r["run_id"], []).append(
                        B.outcome(r, stop_from_scores(pts, sp, T)))
            if collect_extra:
                for r, (pts, X) in zip(test, cache1):
                    with torch.no_grad():
                        s = np.clip(gate.model(torch.tensor(X)).numpy()[:, 0], 0, 100)
                    res["kan_every_epoch"].setdefault(r["run_id"], []).append(
                        B.outcome(r, stop_from_scores(pts, s, T)))
        print(f"[{label}] held out {held} done", flush=True)
    return res, (agree / n_pts if n_pts else None), T


def main():
    t0 = time.time()
    out = {}
    W = load_workloads()
    res, agreement, T = run_lowo(W, "default", collect_extra=True)
    allruns = [r for w in WORKLOADS for r in W[w]]
    res["patience8"] = {r["run_id"]: [B.outcome(r, B.stop_patience(r, 8))] for r in allruns}
    res["patience8_every2"] = {r["run_id"]: [B.outcome(r, patience_at_decisions(r, 8))] for r in allruns}
    res["extrapolation"] = {r["run_id"]: [B.outcome(r, stop_extrapolation(r))] for r in allruns}
    out["policies"] = {k: summarise(v, T) for k, v in res.items() if v}
    out["band_agreement_spline_vs_symbolic"] = agreement
    ids = sorted(res["kan"])
    tests = {}
    for other in ["kan_every_epoch", "kan_spline", "patience8_every2", "extrapolation"]:
        for key in ["regret", "stop_epoch"]:
            a = per_run(res["kan"], key); b = per_run(res[other], key)
            x = [a[i] for i in ids]; y = [b[i] for i in ids]
            tests[f"{other}@{key}"] = (1.0 if all(abs(u - v) < 1e-12 for u, v in zip(x, y))
                                       else float(wilcoxon(x, y, zero_method="zsplit").pvalue))
    out["tests_vs_kan"] = tests

    # feature-constant sensitivity (KAN gate only)
    sens = {}
    orig_th = H.TrainingHistory
    orig_esi = F._epochs_since_improvement
    for window, norm in [(3, 30), (8, 30), (5, 20), (5, 40)]:
        H.TrainingHistory = functools.partial(orig_th, window=window)
        F._epochs_since_improvement = functools.partial(orig_esi, normalise_by=norm)
        Wv = load_workloads()
        r_v, _, _ = run_lowo(Wv, f"w={window},n={norm}", collect_extra=False)
        sens[f"w{window}_n{norm}"] = summarise(r_v["kan"], T)
    H.TrainingHistory, F._epochs_since_improvement = orig_th, orig_esi
    sens["w5_n30"] = out["policies"]["kan"]
    out["reproduces_main_study"] = out["policies"]["kan"]
    out["sensitivity"] = sens
    out["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "seconds"}, indent=1))


if __name__ == "__main__":
    main()
