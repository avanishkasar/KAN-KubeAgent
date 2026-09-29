"""kan_gate/real_run_logger.py must produce exactly the schema
kan_gate.train.load_real_dataset reads back - this is the harvesting half
of training the gate on real runs instead of synthetic ones."""
from kan_gate.real_run_logger import save_decision_points
from kan_gate.train import load_real_dataset

SAMPLE_FEATURES = {
    "loss_plateau_score": 0.1, "gradient_trend": 0.8, "lr_decay_benefit": 0.05,
    "gpu_hours_remaining_vs_budget": 0.9, "epochs_since_improvement": 0.05,
}


def test_round_trips_through_load_real_dataset(tmp_path):
    audit_log = [
        {"trainjob": "demo", "decision": "continue", "gate_score": 12.0, "features": SAMPLE_FEATURES},
        {"trainjob": "demo", "decision": "early_stop", "gate_score": 91.0, "features": SAMPLE_FEATURES},
    ]
    path = save_decision_points("demo-job", audit_log, out_dir=tmp_path)
    assert path is not None and path.exists()

    rows = load_real_dataset(tmp_path)
    assert len(rows) == 2
    assert rows[0][0] == SAMPLE_FEATURES
    assert {d for _, d in rows} == {"continue", "early_stop"}


def test_entries_without_features_are_skipped(tmp_path):
    audit_log = [
        {"trainjob": "demo", "decision": "continue", "gate_score": 12.0, "features": SAMPLE_FEATURES},
        {"trainjob": "demo", "decision": "continue", "gate_score": 12.0},  # pre-feature-logging entry
    ]
    path = save_decision_points("demo-job", audit_log, out_dir=tmp_path)
    rows = load_real_dataset(tmp_path)
    assert len(rows) == 1


def test_empty_or_featureless_audit_log_saves_nothing(tmp_path):
    assert save_decision_points("demo-job", [], out_dir=tmp_path) is None
    assert save_decision_points("demo-job", [{"decision": "continue"}], out_dir=tmp_path) is None
    assert list(tmp_path.glob("*.json")) == []


def test_job_name_is_sanitised_for_the_filename(tmp_path):
    path = save_decision_points("weird/job:name", [
        {"decision": "continue", "features": SAMPLE_FEATURES},
    ], out_dir=tmp_path)
    assert path is not None
    assert "/" not in path.name and ":" not in path.name
