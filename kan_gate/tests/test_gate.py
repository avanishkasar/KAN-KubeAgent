"""Fast unit tests for the KAN gate module - no cluster, no real training run."""
import torch

from kan_gate.features import TrainingHistory, extract_features
from kan_gate.gate import KANGate
from kan_gate.reference_policy import reference_decision
from kan_gate.synthetic import generate_synthetic_history
from kan_gate.train import build_synthetic_dataset, rows_to_tensors


def test_extract_features_shape():
    history = generate_synthetic_history(num_epochs=20, plateau_at_epoch=10, seed=1)
    features = extract_features(history)
    assert set(features.keys()) == {
        "loss_plateau_score",
        "gradient_trend",
        "lr_decay_benefit",
        "gpu_hours_remaining_vs_budget",
        "epochs_since_improvement",
    }
    for value in features.values():
        assert 0.0 <= value <= 1.0


def test_plateaued_run_favours_stop_over_continue():
    plateaued = generate_synthetic_history(num_epochs=25, plateau_at_epoch=5, seed=2)
    healthy = generate_synthetic_history(num_epochs=25, plateau_at_epoch=None, seed=3)

    plateaued_features = extract_features(plateaued)
    healthy_features = extract_features(healthy)

    assert plateaued_features["loss_plateau_score"] > healthy_features["loss_plateau_score"]
    assert reference_decision(plateaued_features) in ("adjust_lr", "early_stop")
    assert reference_decision(healthy_features) == "continue"


def test_gate_trains_and_decides():
    rows = build_synthetic_dataset(n_runs=8, seed=42)
    X, y = rows_to_tensors(rows)

    gate = KANGate()
    gate.fit(X, y, steps=20)  # short run, just checking the pipeline works

    features = extract_features(generate_synthetic_history(num_epochs=20, plateau_at_epoch=8, seed=4))
    result = gate.decide(features)
    assert result.decision in ("continue", "adjust_lr", "early_stop")
    assert 0.0 <= result.score <= 100.0
    assert result.formula.startswith("stop_score =")
