"""Live hardware telemetry - real CPU/RAM/GPU numbers from the machine this
process is running on, sampled on demand.

No fabricated numbers: CPU/RAM come from psutil (a real OS-level read every
call). GPU comes from `nvidia-smi` if it exists on this machine; if it
doesn't (no NVIDIA GPU, or a machine without the driver - this sandbox has
neither), gpu_available is reported False rather than making a number up.
This is what makes the live dashboard portable across machines (a laptop,
this sandbox, the college GPU box) with zero code changes - it detects
what's actually there each time it's asked.
"""
from __future__ import annotations

import shutil
import subprocess

import psutil

_NVIDIA_SMI = shutil.which("nvidia-smi")


def _sample_gpus() -> list[dict]:
    if not _NVIDIA_SMI:
        return []
    try:
        out = subprocess.run(
            [_NVIDIA_SMI,
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0:
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []

    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        idx, name, util, mem_used, mem_total, temp = parts
        gpus.append({
            "index": int(idx), "name": name,
            "utilization_percent": float(util),
            "memory_used_mb": float(mem_used),
            "memory_total_mb": float(mem_total),
            "temperature_c": float(temp),
        })
    return gpus


def sample() -> dict:
    """One real snapshot of this machine's current load. Called on an
    interval by dashboard/backend/live.py while a live run is active."""
    gpus = _sample_gpus()
    vmem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_percent": vmem.percent,
        "memory_used_mb": vmem.used / (1024 * 1024),
        "memory_total_mb": vmem.total / (1024 * 1024),
        "gpu_available": bool(gpus),
        "gpus": gpus,
    }


_torch_cuda_cache: dict = {}


def torch_cuda_available() -> bool | None:
    """Whether the PyTorch that training will use can see a CUDA GPU. Checked
    once in a subprocess (importing torch here would slow the server)."""
    if "v" not in _torch_cuda_cache:
        import sys
        try:
            r = subprocess.run([sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                               capture_output=True, timeout=90)
            _torch_cuda_cache["v"] = r.returncode == 0
        except Exception:
            _torch_cuda_cache["v"] = None
    return _torch_cuda_cache["v"]
