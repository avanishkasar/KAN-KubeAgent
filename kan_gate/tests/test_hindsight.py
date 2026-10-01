from kan_gate.features import FEATURE_NAMES
from kan_gate.hindsight import decision_points, hindsight_score, remaining_gain


def test_no_improvement_left_scores_100():
    curve = [1.0, 0.5, 0.4, 0.45, 0.5, 0.55]  # best at epoch 3, then overfits
    assert remaining_gain(curve, 3) == 0.0
    assert hindsight_score(curve, 3) == 100.0
    assert hindsight_score(curve, 6) == 100.0


def test_large_improvement_left_scores_0():
    curve = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    assert remaining_gain(curve, 2) > 0.05
    assert hindsight_score(curve, 2) == 0.0


def test_score_is_linear_inside_the_tolerance():
    curve = [1.0, 0.99]  # 1% relative improvement left after epoch 1
    assert abs(hindsight_score(curve, 1, tolerance=0.05) - 80.0) < 1e-6


def test_decision_points_are_causal_and_complete():
    curve = [2.0 * 0.9 ** i for i in range(20)]
    grads = [1.0 / (i + 1) for i in range(20)]
    points = decision_points(curve, grads, check_every=2, first_check=4)
    assert [p["epochs_seen"] for p in points] == [4, 6, 8, 10, 12, 14, 16, 18]
    for p in points:
        assert set(p["features"]) == set(FEATURE_NAMES)
        assert all(0.0 <= v <= 1.0 for v in p["features"].values())
        assert 0.0 <= p["target"] <= 100.0
