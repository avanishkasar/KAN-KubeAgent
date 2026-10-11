"""The learning-rate adjustment guard in agents/executor.py."""
from agents.executor import KANGatedExecutor


class _FakeStatus:
    def __init__(self, lr):
        self.lr = lr


class _FakeClient:
    def __init__(self):
        self.lr = 1e-3
        self.stopped = False

    def get_status(self, name):
        return _FakeStatus(self.lr)

    def patch_lr(self, name, lr):
        self.lr = lr

    def stop(self, name):
        self.stopped = True


def _outcomes(executor, decisions):
    return [executor.execute("job", d, 50.0, "f")["outcome"] for d in decisions]


def test_guard_spaces_out_and_caps_adjustments():
    client = _FakeClient()
    ex = KANGatedExecutor(client, max_lr_adjustments=3, cooldown_checks=2)
    out = _outcomes(ex, ["adjust_lr"] * 12)
    assert out[:7] == ["LR_ADJUSTED", "LR_ADJUST_DEFERRED", "LR_ADJUST_DEFERRED",
                       "LR_ADJUSTED", "LR_ADJUST_DEFERRED", "LR_ADJUST_DEFERRED", "LR_ADJUSTED"]
    assert out.count("LR_ADJUSTED") == 3
    assert abs(client.lr - 1e-3 / 8) < 1e-12


def test_continue_decisions_count_towards_cooldown():
    client = _FakeClient()
    ex = KANGatedExecutor(client, max_lr_adjustments=3, cooldown_checks=2)
    out = _outcomes(ex, ["adjust_lr", "continue", "continue", "adjust_lr"])
    assert out == ["LR_ADJUSTED", "NO_OP", "NO_OP", "LR_ADJUSTED"]


def test_guard_can_be_disabled():
    client = _FakeClient()
    ex = KANGatedExecutor(client, max_lr_adjustments=None, cooldown_checks=0)
    assert _outcomes(ex, ["adjust_lr"] * 5) == ["LR_ADJUSTED"] * 5


def test_jobs_are_tracked_separately_and_stop_still_works():
    client = _FakeClient()
    ex = KANGatedExecutor(client)
    assert ex.execute("a", "adjust_lr", 50.0, "f")["outcome"] == "LR_ADJUSTED"
    assert ex.execute("b", "adjust_lr", 50.0, "f")["outcome"] == "LR_ADJUSTED"
    assert ex.execute("a", "early_stop", 90.0, "f")["outcome"] == "EARLY_STOPPED"
    assert client.stopped
