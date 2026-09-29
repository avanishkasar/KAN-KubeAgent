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

Priority alone only matters when something else is *competing* for the
same cores - on an otherwise-idle machine, "nice" won't stop training
from showing high CPU%, because there's nothing to yield to. To actually
keep background mode's footprint small and predictable (visible single-
digit percentages, not "however much happens to be free"), this module
also restricts the process to a fraction of the machine's logical cores
via CPU affinity - a hard cap, not a suggestion, honoured directly by the
OS scheduler on both Windows and Linux.
"""
from __future__ import annotations

import psutil

MODES = ("background", "normal", "turbo")

# Background mode is capped to this fraction of logical cores (min 1) so
# it can't saturate the machine even when nothing else is running.
_BACKGROUND_CORE_FRACTION = 0.25


def apply_mode(pid: int, mode: str) -> str:
    """Best-effort priority + core-affinity adjustment. Returns a
    human-readable outcome string for the live event log - never raises,
    since this is a nice-to-have, not something that should fail a
    training run."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode!r}, expected one of {MODES}")

    outcomes = []
    try:
        proc = psutil.Process(pid)
        if psutil.WINDOWS:
            targets = {
                "background": psutil.IDLE_PRIORITY_CLASS,
                "normal": psutil.NORMAL_PRIORITY_CLASS,
                "turbo": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
            }
            proc.nice(targets[mode])
            outcomes.append(f"priority='{mode}'")
        else:
            # Standard POSIX niceness: higher = lower priority. Raising it
            # (background) never needs privileges; lowering it (turbo)
            # usually does, so we try and fall back quietly if denied -
            # turbo still runs, just at the default OS priority.
            targets = {"background": 15, "normal": 0, "turbo": -5}
            try:
                proc.nice(targets[mode])
                outcomes.append(f"priority='{mode}'")
            except psutil.AccessDenied:
                if mode == "turbo":
                    outcomes.append(
                        "priority elevation denied (not privileged enough) - running at "
                        "normal OS priority"
                    )
                else:
                    raise

        try:
            total_cores = psutil.cpu_count(logical=True) or 1
            if mode == "background":
                capped = max(1, int(total_cores * _BACKGROUND_CORE_FRACTION))
                proc.cpu_affinity(list(range(capped)))
                outcomes.append(f"capped to {capped}/{total_cores} CPU cores")
            else:
                proc.cpu_affinity(list(range(total_cores)))
        except (AttributeError, NotImplementedError):
            pass  # cpu_affinity isn't available on this platform (e.g. macOS) - priority alone still applies

        return f"{', '.join(outcomes)} (pid {pid})" if outcomes else f"applied '{mode}' mode (pid {pid})"
    except Exception as exc:  # pragma: no cover - best-effort, must never crash a run
        return f"could not fully apply '{mode}' mode ({exc.__class__.__name__}): continuing at default settings"
