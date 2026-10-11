"""Numbers, tables and figures for the MethodsX article (paper/methodsx/).

Everything in the article that is a measured value comes from this script,
which reads only experiment outputs on disk:

    experiments/results/generalization.json     (experiments/generalization.py)
    experiments/results/closed_loop*.json       (experiments/closed_loop.py)
    experiments/data/curves*.jsonl              (experiments/collect_curves.py)
    kan_gate/checkpoints/kan_gate               (python -m kan_gate.train --curves ...)

It writes paper/methodsx/generated/numbers.tex (LaTeX macros), the table
bodies in paper/methodsx/generated/, and the figures in
paper/methodsx/figures/. No number is typed into the article by hand.

    python -m experiments.paper_methodsx
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
import numpy as np
import sympy
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from experiments import benchmark as B  # noqa: E402
from experiments.collect_curves import WORKLOADS, curves_path  # noqa: E402
from kan_gate.features import FEATURE_NAMES  # noqa: E402
from kan_gate.gate import DEFAULT_CKPT_PATH, KANGate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = REPO_ROOT / "experiments" / "results"
PAPER = REPO_ROOT / "paper" / "methodsx"
GEN = PAPER / "generated"
FIG = PAPER / "figures"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e1e0d9", "#898781"
LR_RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # light -> dark

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "axes.titlesize": 8, "axes.titlecolor": INK,
})

OURS = "KAN gate (hindsight labels)"
POLICIES = [  # macro key, name in results, label in tables
    ("kan", OURS, "KAN gate, hindsight labels (ours)"),
    ("lin", "Linear (hindsight labels)", "Linear, hindsight labels"),
    ("mlp", "MLP (hindsight labels)", "MLP, hindsight labels"),
    ("tree", "Tree depth-3 (hindsight labels)", "Tree (depth 3), hindsight labels"),
    ("kanrule", "KAN gate (rule labels)", "KAN gate, rule labels"),
    ("rule", "Reference rule", "Hand-written reference rule"),
    ("patthree", "Patience-3", "Patience 3"),
    ("patfive", "Patience-5", "Patience 5"),
    ("pateight", "Patience-8", "Patience 8"),
    ("median", "Median stopping rule", "Median stopping rule"),
    ("oracle", "Hindsight oracle", "Hindsight oracle (not realisable)"),
    ("noes", "No early stopping", "No early stopping"),
]
KEY_OF = {name: key for key, name, _ in POLICIES}
LABEL_OF = {name: label for _, name, label in POLICIES}
WL_KEY = {"fmnist_cnn": "fmnist", "mnist_cnn": "mnist", "cifar10_cnn": "cifar",
          "fmnist_mlp_sgd": "mlpsgd"}
WL_SHORT = {"fmnist_cnn": "Fashion-MNIST CNN", "mnist_cnn": "MNIST CNN",
            "cifar10_cnn": "CIFAR-10 CNN", "fmnist_mlp_sgd": "Fashion-MNIST MLP (SGD)"}
FEATURE_SYMBOL = {n: f"x_{i + 1}" for i, n in enumerate(FEATURE_NAMES)}

macros: list[str] = []


def m(name: str, value) -> None:
    macros.append(f"\\expandafter\\def\\csname {name}\\endcsname{{{value}}}")


def f1(x): return f"{x:.1f}"
def f2(x): return f"{x:.2f}"
def pv(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- offline results

def offline(g: dict) -> None:
    cfg = g["config"]
    m("nRuns", cfg["n_runs"])
    m("nWorkloads", len(cfg["workloads"]))
    m("budget", cfg["budget_epochs"])
    m("nBoot", f"{cfg['n_boot']:,}".replace(",", "{,}"))
    m("nDecisionPoints", f"{sum(w['decision_points'] for w in g['workloads'].values()):,}".replace(",", "{,}"))
    m("nEpochsRecorded", f"{cfg['n_runs'] * cfg['budget_epochs']:,}".replace(",", "{,}"))

    for proto in ("within", "lowo"):
        for name, s in g["summary"][proto].items():
            k = KEY_OF[name]
            pre = f"R@{proto}@{k}@"
            m(pre + "saved", f1(s["compute_saved_pct"]))
            m(pre + "savedlo", f1(s["compute_saved_ci"][0]))
            m(pre + "savedhi", f1(s["compute_saved_ci"][1]))
            m(pre + "regret", f2(s["mean_regret_pct"]))
            m(pre + "regretlo", f2(s["mean_regret_ci"][0]))
            m(pre + "regrethi", f2(s["mean_regret_ci"][1]))
            m(pre + "prem", f1(s["premature_rate_pct"]))
            m(pre + "p90", f2(s["p90_regret_pct"]))
            m(pre + "max", f1(s["max_regret_pct"]))
            m(pre + "acc", f2(s["mean_acc_drop_pp"]))
        for w, pol in g["per_workload"][proto].items():
            for name, s in pol.items():
                pre = f"W@{proto}@{WL_KEY[w]}@{KEY_OF[name]}@"
                m(pre + "saved", f1(s["compute_saved_pct"]))
                m(pre + "regret", f2(s["mean_regret_pct"]))
                m(pre + "prem", f1(s["premature_rate_pct"]))
        for metric, comp in g["tests"][proto].items():
            mk = "regret" if metric == "regret" else "stop"
            for other, t in comp.items():
                pre = f"T@{proto}@{KEY_OF[other]}@{mk}@"
                m(pre + "p", pv(t["p_holm"]))
                m(pre + "diff", f2(t["median_diff"]) if mk == "regret" else f1(t["median_diff"]))

    for w, s in g["workloads"].items():
        pre = f"D@{WL_KEY[w]}@"
        m(pre + "acc", f1(100 * s["best_val_acc_median"]))
        m(pre + "accmax", f1(100 * s["best_val_acc_max"]))
        m(pre + "bestep", f1(s["best_epoch_median"]))
        m(pre + "late", f1(100 * s["frac_best_in_last_5"]))
        m(pre + "sec", f2(s["epoch_seconds_mean"]))

    it = g["interpretability"]
    m("nGates", it["n_gates"])
    m("faithMax", f"{it['faithfulness_max_abs']:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}")
    m("faithMed", f"{it['faithfulness_median']:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}")
    m("termsMed", f"{it['formula_terms_median']:g}")
    m("termsMax", f"{it['formula_terms_max']:g}")
    m("edgesMed", f"{it['active_edges_median']:g}")
    m("maeSpline", f2(it["val_mae_spline_mean"]))
    m("maeSymbolic", f2(it["val_mae_symbolic_mean"]))
    m("fitSeconds", f1(it["train_seconds_mean"]))
    lat = json.loads((RES / "results.json").read_text())["latency_us"]
    m("latKan", f1(lat["KAN (hindsight labels, ours)"] / 1000))
    m("latLin", f2(lat["Linear (hindsight labels)"] / 1000))

    for a in g["ablation"]:
        key = "none" if a["removed_feature"] is None else f"x{FEATURE_NAMES.index(a['removed_feature']) + 1}"
        m(f"A@{key}@saved", f1(a["compute_saved_pct"]))
        m(f"A@{key}@regret", f2(a["mean_regret_pct"]))
        m(f"A@{key}@prem", f1(a["premature_rate_pct"]))
    for t in g["tau_sweep"]:
        key = f"{round(100 * t['tau'])}"
        m(f"S@{key}@saved", f1(t["compute_saved_pct"]))
        m(f"S@{key}@regret", f2(t["mean_regret_pct"]))
        m(f"S@{key}@prem", f1(t["premature_rate_pct"]))


def table_workloads(g: dict) -> None:
    rows = []
    for w, s in g["workloads"].items():
        args, lrs = WORKLOADS[w]
        ds = {"fmnist_cnn": "Fashion-MNIST~\\cite{xiao2017fashion}", "mnist_cnn": "MNIST~\\cite{lecun1998mnist}",
              "cifar10_cnn": "CIFAR-10~\\cite{krizhevsky2009cifar}",
              "fmnist_mlp_sgd": "Fashion-MNIST~\\cite{xiao2017fashion}"}[w]
        model = {"fmnist_cnn": "CNN, 2 conv", "mnist_cnn": "CNN, 2 conv", "cifar10_cnn": "CNN, 3 conv",
                 "fmnist_mlp_sgd": "MLP 256--128"}[w]
        opt = "SGD, mom.\\ 0.9" if "sgd" in args else "Adam"
        lr = f"$10^{{{int(np.log10(min(lrs)))}}}$--${max(lrs):g}$".replace("$0.3$", "$0.3$")
        rows.append(f"{ds} & {model} & {opt} & {lr} & {100 * s['best_val_acc_median']:.1f} & "
                    f"{s['best_epoch_median']:.1f} & {100 * s['frac_best_in_last_5']:.1f} & "
                    f"{s['epoch_seconds_mean']:.2f} \\\\")
    (GEN / "table_workloads.tex").write_text("\n".join(rows) + "\n")


def table_main(g: dict) -> None:
    rows = []
    for key, name, label in POLICIES:
        cells = []
        for proto in ("within", "lowo"):
            s = g["summary"][proto][name]
            cells.append(f"{f1(s['compute_saved_pct'])} [{f1(s['compute_saved_ci'][0])}, {f1(s['compute_saved_ci'][1])}]")
            cells.append(f"{f2(s['mean_regret_pct'])} [{f2(s['mean_regret_ci'][0])}, {f2(s['mean_regret_ci'][1])}]")
            cells.append(f1(s["premature_rate_pct"]))
        line = f"{label} & " + " & ".join(cells) + " \\\\"
        if key == "kan":
            line = "\\rowcolor{kblue!8} " + line
        rows.append(line)
        if key in ("kanrule", "median"):
            rows.append("\\addlinespace[2pt]")
    (GEN / "table_main.tex").write_text("\n".join(rows) + "\n")


def table_per_workload(g: dict) -> None:
    rows = []
    for key in ("kan", "lin", "mlp", "patfive", "pateight", "oracle"):
        name = next(n for k, n, _ in POLICIES if k == key)
        cells = []
        for w in g["workloads"]:
            s = g["per_workload"]["lowo"][w][name]
            cells.append(f"{f1(s['compute_saved_pct'])} / {f2(s['mean_regret_pct'])}")
        line = f"{LABEL_OF[name]} & " + " & ".join(cells) + " \\\\"
        if key == "kan":
            line = "\\rowcolor{kblue!8} " + line
        rows.append(line)
    (GEN / "table_per_workload.tex").write_text("\n".join(rows) + "\n")


def table_tests(g: dict) -> None:
    rows = []
    for key, name, label in POLICIES:
        if key in ("kan", "oracle", "noes"):
            continue
        cells = []
        for proto in ("within", "lowo"):
            t = g["tests"][proto]
            cells += [f1(t["stop_epoch"][name]["median_diff"]), pv(t["stop_epoch"][name]["p_holm"]),
                      f2(t["regret"][name]["median_diff"]), pv(t["regret"][name]["p_holm"])]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    (GEN / "table_tests.tex").write_text("\n".join(rows) + "\n")


def table_ablation(g: dict) -> None:
    rows = []
    for a in g["ablation"]:
        f = a["removed_feature"]
        label = "None (all five features)" if f is None else \
            f"${FEATURE_SYMBOL[f]}$ ({f.replace('_', ' ')})".replace("gpu hours remaining vs budget", "budget remaining")
        rows.append(f"{label} & {f1(a['compute_saved_pct'])} & {f2(a['mean_regret_pct'])} & "
                    f"{f1(a['premature_rate_pct'])} \\\\")
    rows.append("\\midrule")
    for t in g["tau_sweep"]:
        rows.append(f"Tolerance $\\tau={100 * t['tau']:g}\\%$ & {f1(t['compute_saved_pct'])} & "
                    f"{f2(t['mean_regret_pct'])} & {f1(t['premature_rate_pct'])} \\\\")
    (GEN / "table_ablation.tex").write_text("\n".join(rows) + "\n")


# ----------------------------------------------------------------------------- closed loop

CLOSED = [  # key, file, label
    ("fmnistnoguard", "closed_loop.json", "Fashion-MNIST CNN, within-workload gate, no guard"),
    ("fmnistguard", "closed_loop_fmnist_cnn_within_guard.json", "Fashion-MNIST CNN, within-workload gate, guard"),
    ("lowofmnist", "closed_loop_fmnist_cnn_lowo_guard.json", "Fashion-MNIST CNN"),
    ("lowomnist", "closed_loop_mnist_cnn_lowo_guard.json", "MNIST CNN"),
    ("lowocifar", "closed_loop_cifar10_cnn_lowo_guard.json", "CIFAR-10 CNN"),
    ("lowomlpsgd", "closed_loop_fmnist_mlp_sgd_lowo_guard.json", "Fashion-MNIST MLP (SGD)"),
    ("cifarnoguard", "closed_loop_cifar10_cnn_lowo_noguard.json", "CIFAR-10 CNN, no guard"),
]


def closed_summary(summ: list[dict], runs: list[dict]) -> dict:
    used = sum(s["epochs_run"] for s in summ)
    budget = sum(s["epochs_budget"] for s in summ)
    rel = [s["rel_change_pct"] for s in summ]
    a = [s["best_val_controlled"] for s in summ]
    b = [s["best_val_uncontrolled"] for s in summ]
    p = 1.0 if all(abs(x - y) < 1e-12 for x, y in zip(a, b)) else float(
        wilcoxon(a, b, zero_method="zsplit").pvalue)
    adj = [s["n_adjust_lr"] for s in summ]
    return {
        "n": len(summ), "used": used, "budget": budget, "saved": 100 * (1 - used / budget),
        "rel_median": statistics.median(rel), "rel_mean": statistics.mean(rel),
        "rel_max": max(rel), "rel_min": min(rel),
        "worse2": sum(r > 2.0 for r in rel), "better2": sum(r < -2.0 for r in rel),
        "p": p, "adj_total": sum(adj), "adj_max": max(adj),
        "deferred": sum(s.get("n_adjust_deferred", 0) for s in summ),
        "stopped": sum(s["stopped_early"] for s in summ),
        "acc_mean_pp": statistics.mean(100 * (s["acc_controlled"] - s["acc_uncontrolled"]) for s in summ),
        "prefix_ok": all(s["identical_prefix_epochs"] >= 1 for s in summ),
    }


def closed_loop() -> dict:
    out = {}
    pooled_summ, pooled_runs = [], []
    for key, fname, label in CLOSED:
        path = RES / fname
        if not path.exists():
            raise SystemExit(f"missing {path}: run experiments/closed_loop.py first")
        d = json.loads(path.read_text())
        cs = closed_summary(d["summary"], d["runs"])
        cs.update({"label": label, "data": d})
        out[key] = cs
        if key.startswith("lowo"):
            pooled_summ += d["summary"]
            pooled_runs += d["runs"]
    out["lowoall"] = {**closed_summary(pooled_summ, pooled_runs), "label": "All four workloads"}

    rows_lowo, rows_guard = [], []
    for key, cs in out.items():
        pre = f"C@{key}@"
        m(pre + "used", cs["used"]); m(pre + "budget", cs["budget"]); m(pre + "n", cs["n"])
        m(pre + "saved", f1(cs["saved"])); m(pre + "relmed", f2(cs["rel_median"]))
        m(pre + "relmean", f2(cs["rel_mean"])); m(pre + "relmax", f1(cs["rel_max"]))
        m(pre + "relmin", f1(cs["rel_min"]))
        m(pre + "worse", cs["worse2"]); m(pre + "better", cs["better2"]); m(pre + "p", pv(cs["p"]))
        m(pre + "adj", cs["adj_total"]); m(pre + "adjmax", cs["adj_max"]); m(pre + "deferred", cs["deferred"])
        m(pre + "stopped", cs["stopped"]); m(pre + "acc", f2(cs["acc_mean_pp"]))
        row = (f"{cs['label']} & {cs['n']} & {cs['used']}/{cs['budget']} & {f1(cs['saved'])} & "
               f"{f2(cs['rel_median'])} & {f1(cs['rel_min'])} / {f1(cs['rel_max'])} & "
               f"{cs['worse2']} & {pv(cs['p'])} & {cs['adj_total']} ({cs['adj_max']}) \\\\")
        if key.startswith("lowo"):
            if key == "lowoall":
                row = "\\midrule\n" + row
            rows_lowo.append(row)
        else:
            rows_guard.append(row)
    order = ["fmnistnoguard", "fmnistguard", "cifarnoguard"]
    rows_guard = [r for k in order for r in rows_guard if r.startswith(out[k]["label"] + " &")]
    rows_guard.insert(2, "\\addlinespace[2pt]")
    rows_guard.append(f"{out['lowocifar']['label']}, guard & " + rows_lowo[2].split(" & ", 1)[1])
    (GEN / "table_closed_lowo.tex").write_text("\n".join(rows_lowo) + "\n")
    (GEN / "table_closed_guard.tex").write_text("\n".join(rows_guard) + "\n")
    return out


# ----------------------------------------------------------------------------- deployed gate

def deployed_gate() -> KANGate:
    gate = KANGate.load(DEFAULT_CKPT_PATH)
    rhs = gate.formula().split("=", 1)[1].strip()
    expr = sympy.sympify(rhs)
    runs = [r for w in WORKLOADS for r in B.load_runs(curves_path(w))]
    X = np.concatenate([r["X"] for r in runs])
    info = B.kan_interpretability(B.KANScorer(gate, "deployed"), X)
    m("depFaith", f"{info['faithfulness_max_abs']:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}")
    m("depTerms", len(sympy.Add.make_args(sympy.expand(expr))))
    m("depRows", f"{len(X):,}".replace(",", "{,}"))
    subs = {sympy.Symbol(n): sympy.Symbol(FEATURE_SYMBOL[n]) for n in FEATURE_NAMES}
    shown = sympy.expand(expr.xreplace(subs))
    shown = shown.xreplace({n: sympy.Float(round(float(n), 2), 4) for n in shown.atoms(sympy.Float)})
    order = [sympy.Symbol(f"x_{i}") for i in range(1, 6)]
    latex = sympy.latex(shown, order="lex", symbol_names={s: str(s) for s in order})
    latex = latex.replace("1.0 ", "").replace("\\left(", "(").replace("\\right)", ")")
    (GEN / "formula_deployed.tex").write_text(latex + "\n")
    (GEN / "formula_deployed_full.txt").write_text(gate.formula() + "\n")
    return gate


# ----------------------------------------------------------------------------- figures

def fig_curves(g: dict) -> None:
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
    handles = []
    for ax, w in zip(axes.flat, g["workloads"]):
        ids = sorted((i for i, c in g["curves"].items()
                      if c["workload"] == w and c["batch_size"] == 64 and c["seed"] == 2),
                     key=lambda i: g["curves"][i]["lr"])
        for color, i in zip(LR_RAMP, ids):
            c = g["curves"][i]
            xs = range(1, len(c["val_loss"]) + 1)
            ln, = ax.plot(xs, c["val_loss"], color=color, lw=1.4, label=f"learning rate {c['lr']:g}")
            if ax is axes.flat[0]:
                handles.append(ln)
            stop = int(np.round(np.mean(g["per_run"]["lowo"][OURS][i]["stop_epoch"])))
            if stop < len(c["val_loss"]):
                ax.plot([stop], [c["val_loss"][stop - 1]], marker="o", ms=5, mfc=color, mec="white",
                        mew=1.2, zorder=5)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(WL_SHORT[w], loc="left")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation loss")
    stop = Line2D([], [], marker="o", ms=5, mfc=GRAY, mec="white", ls="none",
                  label="stopped by KAN gate")
    fig.legend(handles=handles + [stop], loc="lower center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "curves")


def fig_frontier(g: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3), sharey=True)
    lines = [(OURS, BLUE, "KAN gate (ours)"), ("Linear (hindsight labels)", ORANGE, "Linear"),
             ("MLP (hindsight labels)", AQUA, "MLP")]
    titles = {"within": "Within-workload (seed-grouped 3-fold CV)", "lowo": "Leave-one-workload-out"}
    for ax, proto in zip(axes, ("within", "lowo")):
        pat = g["frontiers"]["patience"]
        ax.plot([p["regret"] for p in pat], [p["saved"] for p in pat], color=YELLOW, lw=1.6, ls="--",
                marker="s", ms=3, label="Patience, 1 to 20 epochs")
        for p in pat:
            if p["patience"] in (3, 5, 8) and p["regret"] <= 3.0:
                ax.annotate(f"p={p['patience']}", (p["regret"], p["saved"]), textcoords="offset points",
                            xytext=(3, 5), fontsize=6, color=INK2)
        for name, color, label in lines:
            pts = g["frontiers"][proto][name]
            ax.plot([p["regret"] for p in pts], [p["saved"] for p in pts], color=color, lw=1.8,
                    marker="o", ms=3.5, label=f"{label}, stop threshold 55 to 95")
            d = next(p for p in pts if p["threshold"] == 75)
            ax.plot([d["regret"]], [d["saved"]], marker="o", ms=8, mfc=color, mec="white", mew=1.5, zorder=6)
        o = g["summary"][proto]["Hindsight oracle"]
        ax.plot([o["mean_regret_pct"]], [o["compute_saved_pct"]], marker="*", ms=11, color=INK,
                ls="none", label="Hindsight oracle")
        r = g["summary"][proto]["KAN gate (rule labels)"]
        ax.plot([r["mean_regret_pct"]], [r["compute_saved_pct"]], marker="D", ms=5, color=VIOLET,
                ls="none", label="KAN gate, rule labels")
        ax.set_xlim(-0.05, 3.0)
        ax.set_title(titles[proto], loc="left")
        ax.set_xlabel("mean regret in best validation loss (%)")
    axes[0].set_ylabel("compute saved (% of epochs)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "frontier")


def fig_edges(gate: KANGate) -> None:
    from kan_gate.introspect import network

    net = network(gate)
    edges = [e for e in net["edges"] if e["active"]]
    edges.sort(key=lambda e: (e["layer"], e["j"], e["i"]))
    n = len(edges)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.8, 1.45 * rows + 0.2), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, e in zip(axes.flat, edges):
        color = BLUE if e["layer"] == 0 else ORANGE
        ax.plot(e["curve"]["x"], e["curve"]["y"], color=color, lw=1.8)
        src = f"${FEATURE_SYMBOL[FEATURE_NAMES[e['i']]]}$" if e["layer"] == 0 else f"$h_{e['i'] + 1}$"
        dst = f"$h_{e['j'] + 1}$" if e["layer"] == 0 else "$s$"
        ax.set_title(f"{src} $\\rightarrow$ {dst}", loc="left", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, "edges")
    m("depActiveEdges", n)


def fig_closed(cl: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax = axes[0]
    keys = ["lowofmnist", "lowomnist", "lowocifar", "lowomlpsgd"]
    colors = [BLUE, ORANGE, AQUA, VIOLET]
    for k, c in zip(keys, colors):
        s = cl[k]["data"]["summary"]
        ax.scatter([x["epochs_run"] for x in s], [x["rel_change_pct"] for x in s], s=22, color=c,
                   edgecolor="white", linewidth=0.8, label=cl[k]["label"], zorder=3)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("epochs used of 40")
    ax.set_ylabel("change in best val. loss vs.\nuncontrolled run (%)")
    ax.set_title("Unseen-workload gates, with guard", loc="left")
    ax.legend(fontsize=6, loc="upper left")
    ax = axes[1]
    pairs = [("fmnistnoguard", "fmnistguard", "Fashion-MNIST\nwithin-workload gate"),
             ("cifarnoguard", "lowocifar", "CIFAR-10\nunseen-workload gate")]
    xs = np.arange(len(pairs))
    for off, idx, color, label in ((-0.18, 0, GRAY, "no guard"), (0.18, 1, BLUE, "guard")):
        vals = [[x["n_adjust_lr"] for x in cl[p[idx]]["data"]["summary"]] for p in pairs]
        worst = [cl[p[idx]]["rel_max"] for p in pairs]
        ax.bar(xs + off, [sum(v) for v in vals], width=0.34, color=color, label=f"{label}: LR halvings")
        for x, wv in zip(xs + off, worst):
            ax.annotate(f"worst {wv:+.1f}%", (x, 0), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=6, color="white" if color == BLUE else INK, rotation=90,
                        va="bottom")
    ax.set_xticks(xs, [p[2] for p in pairs])
    ax.set_ylabel("learning-rate halvings (12 runs)")
    ax.set_title("Effect of the learning-rate guard", loc="left")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "closed_loop")


def graphical_abstract(g: dict, cl: dict) -> None:
    """Elsevier graphical abstract: one panel, readable at small size."""
    fig = plt.figure(figsize=(6.6, 2.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)

    def box(x, y, w, h, title, body, edge=INK2, face="white"):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                                       linewidth=1.2, edgecolor=edge, facecolor=face))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 8.4, body, ha="center", va="top", fontsize=7, color=INK2, linespacing=1.35)

    box(1.5, 13, 21, 22, "Real training runs", f"{g['config']['n_runs']} uncontrolled runs\n4 workloads\nval. loss + grad. norm")
    box(27.5, 13, 21, 22, "Hindsight labels", "label each decision by\nthe improvement that\nwas still to come",
        edge=BLUE, face="#eef4fc")
    box(53.5, 13, 21, 22, "Symbolic KAN gate", "KAN [5,3,1], edges fixed\nto x, x², x³, sin\nformula = decision",
        edge=BLUE, face="#eef4fc")
    box(79.5, 13, 19, 22, "Binding control", "continue / halve LR /\nstop, with an exact\nper-feature audit")
    for x0, x1 in ((23.3, 27.0), (49.3, 53.0), (75.3, 79.0)):
        ax.annotate("", xy=(x1, 24), xytext=(x0, 24), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
    lo = g["summary"]["lowo"][OURS]
    ax.text(50, 6.5,
            f"Unseen workloads: {lo['compute_saved_pct']:.1f}% of epochs saved at {lo['mean_regret_pct']:.2f}% mean regret"
            f"   |   live closed loop: {cl['lowoall']['saved']:.1f}% saved",
            ha="center", va="center", fontsize=7.6, color=INK)
    fig.savefig(PAPER / "graphical_abstract.pdf")
    fig.savefig(PAPER / "graphical_abstract.png", dpi=300)
    plt.close(fig)


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    g = json.loads((RES / "generalization.json").read_text())
    offline(g)
    table_workloads(g)
    table_main(g)
    table_per_workload(g)
    table_tests(g)
    table_ablation(g)
    cl = closed_loop()
    gate = deployed_gate()
    fig_curves(g)
    fig_frontier(g)
    fig_edges(gate)
    fig_closed(cl)
    graphical_abstract(g, cl)
    header = ["% Generated by experiments/paper_methodsx.py - do not edit by hand.",
              "\\makeatletter"]
    (GEN / "numbers.tex").write_text("\n".join(header + macros + ["\\makeatother"]) + "\n")
    print(f"wrote {len(macros)} macros, tables and figures to {PAPER}")


if __name__ == "__main__":
    main()
