"""kan_gate/real_run_logger.py must store what kan_gate.train.load_real_dataset
needs to hindsight-label a run: the full monitored curve, not just the
gate's own decisions (re-learning those would only copy the gate)."""
from kan_gate.real_run_logger import save_decision_points
from kan_gate.train import load_real_dataset

SAMPLE_FEATURES = {
    "loss_plateau_score": 0.1, "gradient_trend": 0.8, "lr_decay_benefit": 0.05,
    "gpu_hours_remaining_vs_budget": 0.9, "epochs_since_improvement": 0.05,
}
CURVE = [1.0, 0.8, 0.65, 0.55, 0.5, 0.47, 0.46, 0.455, 0.452, 0.451, 0.452, 0.453]
GRADS = [3.0, 2.8, 2.5, 2.2, 2.0, 1.9, 1.8, 1.7, 1.7, 1.6, 1.6, 1.6]
AUDIT = [
    {"trainjob": "demo", "decision": "continue", "gate_score": 12.0, "features": SAMPLE_FEATURES},
    {"trainjob": "demo", "decision": "early_stop", "gate_score": 91.0, "features": SAMPLE_FEATURES},
]


def test_completed_run_is_hindsight_labeled(tmp_path):
    path = save_decision_points("demo-job", AUDIT, out_dir=tmp_path,
                                monitored_loss=CURVE, grad_norms=GRADS, phase="Completed")
    assert path is not None and path.exists()

    rows, skipped = load_real_dataset(tmp_path)
    assert skipped == 0
    # decision points at epochs 4, 6, 8, 10 of a 12-epoch run
    assert len(rows) == 4
    targets = [t for _, t in rows]
    assert all(0.0 <= t <= 100.0 for t in targets)
    # later decision points have less improvement left -> higher stop score
    assert targets == sorted(targets)


def test_runs_without_an_observed_future_are_skipped(tmp_path):
    save_decision_points("stopped", AUDIT, out_dir=tmp_path,
                         monitored_loss=CURVE[:6], grad_norms=GRADS[:6], phase="Stopped")
    save_decision_points("no-curve", AUDIT, out_dir=tmp_path, phase="Completed")
    rows, skipped = load_real_dataset(tmp_path)
    assert rows == [] and skipped == 2


def test_entries_without_features_are_not_stored(tmp_path):
    audit = AUDIT[:1] + [{"trainjob": "demo", "decision": "continue", "gate_score": 12.0}]
    path = save_decision_points("demo-job", audit, out_dir=tmp_path,
                                monitored_loss=CURVE, grad_norms=GRADS, phase="Completed")
    import json
    assert len(json.loads(path.read_text())["decision_points"]) == 1


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
