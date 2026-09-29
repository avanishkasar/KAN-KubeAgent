"""Sets a real OS-level scheduling priority on the training subprocess.

This is the actual mechanism behind "background" mode: it does not throttle
CPU by sleeping or polling - it tells the operating system's own scheduler
to prefer your foreground applications over the training process. Windows
and Linux both honour this natively; when you're actively using the
machine, the OS gives your foreground apps CPU first and training gets
whatever's left, then automatically gets more when the machine is idle.
No code here decides that trade-off at runtime - the OS scheduler does,
which is the standard, well-tested way background compute (backups,
folding@home-style workloads, etc.) coexists with interactive use.

GPU compute has no equivalent OS-level "nice" - the NVIDIA driver
time-slices compute between processes on its own. Turbo mode's speedup
instead comes from training/fashion_mnist_cnn.py's --workers and --amp
flags (more dataloader throughput, mixed precision), not from priority.
"""
from __future__ import annotations

import psutil

MODES = ("background", "normal", "turbo")


def apply_mode(pid: int, mode: str) -> str:
    """Best-effort priority adjustment. Returns a human-readable outcome
    string for the live event log - never raises, since priority is a
    nice-to-have, not something that should fail a training run."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode!r}, expected one of {MODES}")

    try:
        proc = psutil.Process(pid)
        if psutil.WINDOWS:
            targets = {
                "background": psutil.IDLE_PRIORITY_CLASS,
                "normal": psutil.NORMAL_PRIORITY_CLASS,
                "turbo": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
            }
            proc.nice(targets[mode])
        else:
            # Standard POSIX niceness: higher = lower priority. Raising it
            # (background) never needs privileges; lowering it (turbo)
            # usually does, so we try and fall back quietly if denied -
            # turbo still runs, just at the default OS priority.
            targets = {"background": 15, "normal": 0, "turbo": -5}
            try:
                proc.nice(targets[mode])
            except psutil.AccessDenied:
                if mode == "turbo":
                    return (
                        "turbo mode requested elevated CPU priority but this process "
                        "isn't privileged enough to grant it - training still runs, "
                        "just at normal OS priority"
                    )
                raise
        return f"set process priority for '{mode}' mode (pid {pid})"
    except Exception as exc:  # pragma: no cover - best-effort, must never crash a run
        return f"could not set process priority for '{mode}' mode ({exc.__class__.__name__}): continuing at default priority"
