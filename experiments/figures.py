"""Paper figures, generated only from experiment outputs on disk:
experiments/results/results.json, experiments/results/closed_loop.json and
the deployed gate checkpoint. Writes PDFs to paper/figures/.

    python -m experiments.figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kan_gate.features import FEATURE_NAMES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = REPO_ROOT / "experiments" / "results"
OUT = REPO_ROOT / "paper" / "figures"
BLUE, ORANGE, AQUA, YELLOW, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#898781"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SHORT = {"loss_plateau_score": "plateau", "gradient_trend": "grad. trend", "lr_decay_benefit": "LR-decay",
         "gpu_hours_remaining_vs_budget": "budget left", "epochs_since_improvement": "since impr."}
OURS = "KAN (hindsight labels, ours)"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
})


def fig_curves(results):
    ex = {e["run_id"]: e for e in results["examples"]}
    seed = max(e["seed"] for e in ex.values())
    picks = [f"lr0.0001_bs256_s{seed}", f"lr0.001_bs64_s{seed}", f"lr0.03_bs64_s{seed}"]
    picks = [p for p in picks if p in ex]
    fig, axes = plt.subplots(1, len(picks), figsize=(7.0, 1.9))
    marks = [("KAN (hindsight labels, ours)", BLUE, "KAN gate (ours)", "-"),
             ("Patience-5", ORANGE, "Patience-5", "--"),
             ("Median stopping rule", YELLOW, "Median rule", ":"),
             ("Hindsight oracle", AQUA, "Hindsight oracle", "-.")]
    for ax, rid in zip(axes, picks):
        e = ex[rid]
        xs = range(1, len(e["val_loss"]) + 1)
        ax.plot(xs, e["val_loss"], color=INK, lw=1.3)
        for k, c, lbl, ls in marks:
            ax.axvline(e["stops"][k], color=c, lw=1.4, ls=ls, label=lbl)
        ax.set_title(f"lr={e['lr']:g}, batch={e['batch_size']}", fontsize=8, color=INK)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("validation loss")
    axes[0].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "curves.pdf"); fig.savefig(OUT / "curves.png", dpi=150)
    plt.close(fig)


OFFSETS = {OURS: (-8, -16, "right"), "Patience-8": (-8, 10, "right"), "MLP (hindsight labels)": (2, 16, "left"),
           "Linear (hindsight labels)": (10, 6, "left"), "Hindsight oracle": (6, -10, "left"),
           "Tree depth-3 (hindsight labels)": (4, 4, "left"), "Patience-5": (4, 4, "left"),
           "Patience-3": (-4, 6, "right")}


def fig_pareto(results):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    keep = ["No early stopping", "Patience-3", "Patience-5", "Patience-8", "Median stopping rule",
            "Reference rule (teacher)", "KAN (teacher labels)", "MLP (hindsight labels)",
            "Linear (hindsight labels)", "Tree depth-3 (hindsight labels)", OURS, "Hindsight oracle"]
    label = {OURS: "KAN, hindsight (ours)", "KAN (teacher labels)": "KAN, teacher",
             "MLP (hindsight labels)": "MLP", "Linear (hindsight labels)": "Linear",
             "Tree depth-3 (hindsight labels)": "Tree", "Reference rule (teacher)": "Rule (teacher)",
             "Median stopping rule": "Median rule", "Hindsight oracle": "Oracle", "No early stopping": "No stop"}
    for p in results["policies"]:
        if p["name"] not in keep:
            continue
        s = p["summary"]
        x, y = s["compute_saved_pct"]["mean"], s["mean_regret_pct"]["mean"]
        ours = p["name"] == OURS
        ax.errorbar(x, y, xerr=s["compute_saved_pct"]["std"], yerr=s["mean_regret_pct"]["std"],
                    fmt="o", ms=6 if ours else 4, color=BLUE if ours else GRAY,
                    mec="white", mew=0.8, elinewidth=0.8, capsize=0, zorder=3 if ours else 2)
        dx, dy, ha = OFFSETS.get(p["name"], (4, 3, "left"))
        ax.annotate(label.get(p["name"], p["name"]), (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.3, color=INK if ours else INK2, ha=ha,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6) if abs(dx) + abs(dy) > 14 else None)
    ax.set_xlabel("compute saved (% of epoch budget)")
    ax.set_ylabel("mean per-run regret (%)")
    ax.set_yscale("symlog", linthresh=1)
    fig.tight_layout()
    fig.savefig(OUT / "pareto.pdf"); fig.savefig(OUT / "pareto.png", dpi=150)
    plt.close(fig)


def fig_edges():
    from kan_gate.gate import KANGate
    from kan_gate.introspect import network

    net = network(KANGate.load())
    edges = sorted((e for e in net["edges"] if e["active"]), key=lambda e: -e["importance"])
    n = len(edges)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 1.25 * rows))
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, e in zip(axes.flat, edges):
        ax.set_visible(True)
        ax.plot(e["curve"]["x"], e["curve"]["y"], color=BLUE, lw=1.4)
        src = SHORT[FEATURE_NAMES[e["i"]]] if e["layer"] == 0 else f"h{e['i']}"
        dst = "score" if e["layer"] == len(net["layers"]) - 2 else f"h{e['j']}"
        ax.set_title(f"{src} → {dst}", fontsize=7, color=INK)
        ax.tick_params(labelsize=5.5)
    fig.tight_layout()
    fig.savefig(OUT / "edges.pdf"); fig.savefig(OUT / "edges.png", dpi=150)
    plt.close(fig)
    return net


def fig_contrib(results):
    from kan_gate.gate import KANGate
    from kan_gate.introspect import explain

    gate = KANGate.load()
    ex = next(e for e in results["examples"] if e["run_id"].startswith("lr0.001_bs64"))
    from kan_gate.hindsight import decision_points
    curves = {json.loads(l)["run_id"]: json.loads(l)
              for l in (REPO_ROOT / "experiments" / "data" / "curves.jsonl").read_text().splitlines() if l.strip()}
    rec = curves[ex["run_id"]]
    pts = decision_points(rec["val_loss"], rec["grad_norm"])
    point = max(pts, key=lambda p: gate.raw_score(p["features"]))
    e = explain(gate, point["features"])
    names = ["baseline"] + [SHORT[f] for f in FEATURE_NAMES] + ["score"]
    vals = [e["baseline"]] + [e["contributions"][f] for f in FEATURE_NAMES]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    run = 0.0
    for k, v in enumerate(vals):
        bottom = 0 if k == 0 else run
        top = v if k == 0 else run + v
        color = GRAY if k == 0 else (BLUE if v >= 0 else RED)
        ax.bar(k, top - bottom, bottom=bottom, color=color, width=0.7)
        ax.text(k, max(top, bottom) + 1.5, f"{v:+.1f}" if k else f"{v:.1f}", ha="center", fontsize=6.3, color=INK2)
        run = top
    status = {"continue": "#0ca30c", "adjust_lr": "#fab219", "early_stop": "#d03b3b"}[e["decision"]]
    ax.bar(len(vals), e["raw_output"], color=status, width=0.7)
    ax.text(len(vals), e["raw_output"] + 1.5, f"{e['raw_output']:.1f}", ha="center", fontsize=6.3, color=INK2)
    for y in (40, 75):
        ax.axhline(y, color=INK2, lw=0.6, ls="--")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=6.3)
    ax.set_ylabel("stop score")
    ax.set_title(f"{ex['run_id']}, after epoch {point['epochs_seen']}: {e['decision']}", fontsize=7, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "contrib.pdf"); fig.savefig(OUT / "contrib.png", dpi=150)
    plt.close(fig)
    return {"run_id": ex["run_id"], "epochs_seen": point["epochs_seen"], **{k: e[k] for k in
            ("baseline", "contributions", "raw_output", "decision", "additive")}}


def fig_closed_loop(closed):
    s = sorted(closed["summary"], key=lambda r: r["run_id"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.0))
    labels = [r["run_id"].rsplit("_s", 1)[0].replace("lr", "").replace("_bs", "/") for r in s]
    xs = range(len(s))
    a1.bar([x - 0.2 for x in xs], [r["epochs_budget"] for r in s], width=0.4, color=GRAY, label="uncontrolled")
    a1.bar([x + 0.2 for x in xs], [r["epochs_run"] for r in s], width=0.4, color=BLUE, label="KAN-gated")
    a1.set_ylabel("epochs trained")
    a2.bar([x - 0.2 for x in xs], [r["best_val_uncontrolled"] for r in s], width=0.4, color=GRAY, label="uncontrolled")
    a2.bar([x + 0.2 for x in xs], [r["best_val_controlled"] for r in s], width=0.4, color=BLUE, label="KAN-gated")
    a2.set_ylabel("best validation loss")
    for ax in (a1, a2):
        ax.set_xticks(list(xs), labels, rotation=55, ha="right", fontsize=5.8)
        ax.set_xlabel("learning rate / batch size")
    a1.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "closed_loop.pdf"); fig.savefig(OUT / "closed_loop.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads((RES / "results.json").read_text())
    fig_curves(results)
    fig_pareto(results)
    net = fig_edges()
    contrib = fig_contrib(results)
    extras = {"deployed_formula": net["formula"], "deployed_layers": net["layers"],
              "deployed_active_edges": sum(e["active"] for e in net["edges"]), "contrib_example": contrib}
    closed_path = RES / "closed_loop.json"
    if closed_path.exists():
        fig_closed_loop(json.loads(closed_path.read_text()))
    (RES / "figure_facts.json").write_text(json.dumps(extras, indent=1))
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
