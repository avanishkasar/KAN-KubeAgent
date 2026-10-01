"""Build a labeled dataset and train the KAN gate.

Three data sources, in order of preference:

  --curves PATH        real learning curves collected by
                       experiments/collect_curves.py, labeled in hindsight
                       (kan_gate/hindsight.py). This is how the shipped
                       checkpoint is trained.
  --real-data-dir DIR  runs harvested from the live dashboard
                       (kan_gate/real_run_logger.py), also hindsight-labeled.
                       Only runs that completed are usable: a run the gate
                       stopped early has no observed future to label from.
  --synthetic          synthetic curves labeled by the hand-written reference
                       policy - the original teacher-student setup, kept as
                       a baseline and a test aid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kan_gate.features import FEATURE_NAMES, TrainingHistory, extract_features
from kan_gate.gate import DEFAULT_CKPT_PATH, KANGate
from kan_gate.hindsight import DEFAULT_TOLERANCE, decision_points
from kan_gate.reference_policy import decision_to_score, reference_decision
from kan_gate.synthetic import generate_synthetic_history


def build_synthetic_dataset(n_runs: int = 120, seed: int = 0):
    """(features, decision) rows from synthetic curves, labeled by the
    reference policy."""
    rows = []
    for i in range(n_runs):
        # Roughly a third of runs never plateau within the horizon (healthy
        # run), the rest plateau at a spread of epochs - including early
        # plateaus, which the epoch-5 decision point below must learn to
        # tell apart from a healthy run's early epochs.
        plateau_at = None if i % 3 == 0 else 3 + (i % 26)
        history = generate_synthetic_history(
            num_epochs=30,
            plateau_at_epoch=plateau_at,
            noise=0.02,
            gpu_hours_budget=10.0,
            seed=seed + i,
        )
        for cutoff in (4, 7, 10, 15, 20, 25, 29):
            partial = TrainingHistory(
                loss_history=history.loss_history[: cutoff + 1],
                grad_norm_history=history.grad_norm_history[: cutoff + 1],
                lr=history.lr,
                gpu_hours_used=history.gpu_hours_used * (cutoff + 1) / 30,
                gpu_hours_budget=history.gpu_hours_budget,
            )
            features = extract_features(partial)
            rows.append((features, reference_decision(features)))
    return rows


def load_curves_dataset(path: Path, tolerance: float = DEFAULT_TOLERANCE,
                        run_filter=None):
    """(features, hindsight score) rows from experiments/collect_curves.py
    output. `run_filter(record) -> bool` selects runs (e.g. a CV fold)."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if run_filter is not None and not run_filter(rec):
            continue
        for p in decision_points(rec["val_loss"], rec["grad_norm"], tolerance=tolerance):
            rows.append((p["features"], p["target"]))
    return rows


def load_real_dataset(data_dir: Path, tolerance: float = DEFAULT_TOLERANCE):
    """(features, hindsight score) rows from runs harvested by
    kan_gate/real_run_logger.py. Returns (rows, skipped_files)."""
    rows, skipped = [], 0
    for f in sorted(data_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        usable = (isinstance(rec, dict) and rec.get("phase") == "Completed"
                  and len(rec.get("monitored_loss", [])) > 4
                  and len(rec.get("grad_norm", [])) == len(rec.get("monitored_loss", [])))
        if not usable:
            skipped += 1
            continue
        for p in decision_points(rec["monitored_loss"], rec["grad_norm"], tolerance=tolerance):
            rows.append((p["features"], p["target"]))
    return rows, skipped


def rows_to_tensors(rows):
    """Targets may be decisions (teacher labels) or 0-100 scores (hindsight)."""
    X = torch.tensor(
        [[f[name] for name in FEATURE_NAMES] for f, _ in rows], dtype=torch.float32
    )
    y = torch.tensor(
        [[decision_to_score(t) if isinstance(t, str) else float(t)] for _, t in rows],
        dtype=torch.float32,
    )
    return X, y


def train_gate(rows, steps: int = 200, seed: int = 42, test_fraction: float = 0.2) -> KANGate:
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(rows), generator=g).tolist()
    rows = [rows[i] for i in order]
    split = int(len(rows) * (1 - test_fraction))
    X_train, y_train = rows_to_tensors(rows[:split])
    X_test, y_test = rows_to_tensors(rows[split:]) if split < len(rows) else (X_train, y_train)
    gate = KANGate(seed=seed)
    gate.fit(X_train, y_train, steps=steps, lamb=0.01, test_X=X_test, test_y=y_test)
    gate.formula()  # fixes every edge to its symbolic form: the formula IS the model
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true",
                     help="Synthetic curves + reference-policy labels (baseline/testing).")
    src.add_argument("--curves", type=Path, default=None,
                     help="Real learning curves (experiments/data/curves.jsonl), hindsight-labeled.")
    parser.add_argument("--real-data-dir", type=Path, default=Path("kan_gate/data/real_runs"),
                        help="Harvested live runs, hindsight-labeled (default source).")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="Relative improvement still worth training for (hindsight labels).")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=Path, default=None)
    args = parser.parse_args()

    if args.synthetic:
        print("[kan_gate.train] Synthetic curves, reference-policy labels (baseline/testing).")
        rows = build_synthetic_dataset()
    elif args.curves:
        rows = load_curves_dataset(args.curves, tolerance=args.tolerance)
        print(f"[kan_gate.train] Real curves from {args.curves}, hindsight labels "
              f"(tolerance={args.tolerance}).")
    else:
        if not args.real_data_dir.exists():
            raise SystemExit(
                f"No harvested runs at {args.real_data_dir}. Pass --curves "
                "experiments/data/curves.jsonl, or --synthetic for the baseline gate."
            )
        rows, skipped = load_real_dataset(args.real_data_dir, tolerance=args.tolerance)
        print(f"[kan_gate.train] Harvested runs, hindsight labels; skipped {skipped} file(s) "
              "that were stopped early, unfinished, or in the old self-labeled format.")

    if not rows:
        raise SystemExit("No labeled decision points found.")
    print(f"[kan_gate.train] {len(rows)} labeled decision points.")
    gate = train_gate(rows, steps=args.steps, seed=args.seed)
    print(f"[kan_gate.train] Learned formula:\n  {gate.formula()}")

    save_path = args.save_path or DEFAULT_CKPT_PATH
    gate.save(save_path)
    print(f"[kan_gate.train] Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
