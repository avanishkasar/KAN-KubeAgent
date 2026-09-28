"""Build a labeled dataset and train the KAN gate.

Default data source is real training-run logs (see --real-data-dir).
Pass --synthetic to generate data instead - a testing/debugging aid only,
per research/proposal/methodology_draft.md Section 6.2, never the primary
path for a result reported in the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kan_gate.features import FEATURE_NAMES, TrainingHistory, extract_features
from kan_gate.gate import KANGate
from kan_gate.reference_policy import decision_to_score, reference_decision
from kan_gate.synthetic import generate_synthetic_history


def build_synthetic_dataset(n_runs: int = 60, seed: int = 0):
    rows = []
    for i in range(n_runs):
        plateau_at = None if i % 3 == 0 else 5 + (i % 20)
        history = generate_synthetic_history(
            num_epochs=30,
            plateau_at_epoch=plateau_at,
            noise=0.02,
            gpu_hours_budget=10.0,
            seed=seed + i,
        )
        # Sample a handful of decision points from each run's trajectory.
        for cutoff in (10, 15, 20, 25, 29):
            partial = TrainingHistory(
                loss_history=history.loss_history[: cutoff + 1],
                grad_norm_history=history.grad_norm_history[: cutoff + 1],
                lr=history.lr,
                gpu_hours_used=history.gpu_hours_used * (cutoff + 1) / 30,
                gpu_hours_budget=history.gpu_hours_budget,
            )
            features = extract_features(partial)
            decision = reference_decision(features)
            rows.append((features, decision))
    return rows


def load_real_dataset(data_dir: Path):
    """Load decision-point rows extracted from real TrainJob logs.

    Expected format: one JSON file per run under data_dir, each a list of
    {"features": {...}, "decision": "continue"|"adjust_lr"|"early_stop"}
    objects, as produced by scripts/extract_decision_points.py.
    """
    rows = []
    for f in sorted(data_dir.glob("*.json")):
        entries = json.loads(f.read_text())
        for entry in entries:
            rows.append((entry["features"], entry["decision"]))
    return rows


def rows_to_tensors(rows):
    X = torch.tensor(
        [[f[name] for name in FEATURE_NAMES] for f, _ in rows], dtype=torch.float32
    )
    y = torch.tensor([[decision_to_score(d)] for _, d in rows], dtype=torch.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true",
                         help="Use the synthetic generator instead of real logs (testing only).")
    parser.add_argument("--real-data-dir", type=Path, default=Path("kan_gate/data/real_runs"),
                         help="Directory of extracted real-run decision points (default path).")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-path", type=Path, default=None)
    args = parser.parse_args()

    if args.synthetic:
        print("[kan_gate.train] Using SYNTHETIC data (debugging/testing mode only).")
        rows = build_synthetic_dataset()
    else:
        if not args.real_data_dir.exists():
            raise SystemExit(
                f"No real training data found at {args.real_data_dir}. "
                "Run agents/ against a real TrainJob first, or pass --synthetic "
                "to test the gate without real data."
            )
        rows = load_real_dataset(args.real_data_dir)

    print(f"[kan_gate.train] {len(rows)} labeled decision points.")
    split = int(len(rows) * 0.8)
    train_rows, test_rows = rows[:split], rows[split:]

    X_train, y_train = rows_to_tensors(train_rows)
    X_test, y_test = rows_to_tensors(test_rows) if test_rows else (X_train, y_train)

    gate = KANGate()
    gate.fit(X_train, y_train, steps=args.steps, lamb=0.01, test_X=X_test, test_y=y_test)

    print(f"[kan_gate.train] Learned formula:\n  {gate.formula()}")

    from kan_gate.gate import DEFAULT_CKPT_PATH
    save_path = args.save_path or DEFAULT_CKPT_PATH
    gate.save(save_path)
    print(f"[kan_gate.train] Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
