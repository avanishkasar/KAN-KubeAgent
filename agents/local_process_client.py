"""Real local TrainJobClient - runs training/fashion_mnist_cnn.py as an
actual subprocess on this machine and reads its real stdout live.

This is the client used by the dashboard's Live mode (dashboard/backend/live.py)
and by k8s/README.md's "no cluster available yet" path: it does real
training (real forward/backward passes, real CPU/GPU load - see
dashboard/backend/hardware.py for how that load is measured), just without
needing Kubernetes. Implements the same TrainJobClient protocol as
MockTrainJobClient and KubeflowTrainJobClient, so agents/graph.py does not
change.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from agents.process_priority import apply_mode
from agents.trainjob_client import TrainJobStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = REPO_ROOT / "training" / "fashion_mnist_cnn.py"

# mode -> dataloader/training-side settings. OS scheduling priority and
# CPU core affinity are applied separately via agents/process_priority.py;
# cpu_threads/throttle_ms here are the training-process-internal half of
# "background" actually being light (torch's own thread pool otherwise
# ignores OS niceness and will use every core it can see). See
# agents/process_priority.py's docstring for why GPU has no equivalent of
# "background" priority.
_MODE_TRAINING_ARGS = {
    "background": {"workers": 0, "amp": False, "cpu_threads": 1, "throttle_ms": 50},
    "normal": {"workers": 2, "amp": False, "cpu_threads": 0, "throttle_ms": 0},
    "turbo": {"workers": 4, "amp": True, "cpu_threads": 0, "throttle_ms": 0},
}


@dataclass
class _RunningJob:
    process: subprocess.Popen
    lr_override_path: Path
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    lr: float = 2e-4
    events: list[str] = field(default_factory=list)
    _events_read_index: int = 0
    total_epochs: int = 30
    gpu_hours_budget: float = 10.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def phase(self) -> str:
        if self.process.poll() is None:
            return "Running"
        return "Completed" if self.process.returncode == 0 else "Stopped"


class LocalProcessTrainJobClient:
    """Real subprocess-backed TrainJobClient. One job at a time per
    instance, which matches the dashboard's single live-session model."""

    def __init__(self):
        self._jobs: dict[str, _RunningJob] = {}

    def create(self, name: str, epochs: int = 30, lr: float = 2e-4,
               batch_size: int = 128, subset_size: int = 6000,
               gpu_hours_budget: float = 10.0, mode: str = "normal",
               gpu_index: int | None = None, seed: int | None = None) -> TrainJobStatus:
        if mode not in _MODE_TRAINING_ARGS:
            raise ValueError(f"Unknown mode: {mode!r}, expected one of {tuple(_MODE_TRAINING_ARGS)}")
        mode_args = _MODE_TRAINING_ARGS[mode]

        lr_override_path = Path(tempfile.gettempdir()) / f"kan-kubeagent-{name}-lr.json"
        lr_override_path.write_text(json.dumps({"lr": lr}))

        env = {**os.environ, "LR_OVERRIDE_FILE": str(lr_override_path), "PYTHONUNBUFFERED": "1"}
        if gpu_index is not None:
            # Restricts which physical GPU CUDA sees for this subprocess -
            # the real mechanism for "which GPU do I train on" when the
            # machine has more than one.
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        cmd = [sys.executable, str(TRAINING_SCRIPT),
               "--epochs", str(epochs), "--lr", str(lr),
               "--batch-size", str(batch_size), "--subset-size", str(subset_size),
               "--workers", str(mode_args["workers"]),
               "--cpu-threads", str(mode_args["cpu_threads"]),
               "--throttle-ms", str(mode_args["throttle_ms"])]
        if mode_args["amp"]:
            cmd.append("--amp")
        if seed is not None:
            cmd += ["--seed", str(seed)]

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            cwd=str(REPO_ROOT), env=env,
        )
        job = _RunningJob(process=process, lr_override_path=lr_override_path,
                           lr=lr, total_epochs=epochs, gpu_hours_budget=gpu_hours_budget)
        self._jobs[name] = job
        job.events.append(f"Mode: {mode} - {apply_mode(process.pid, mode)}")

        thread = threading.Thread(target=self._pump_output, args=(name, job), daemon=True)
        thread.start()
        return self.get_status(name)

    def _pump_output(self, name: str, job: _RunningJob) -> None:
        for line in job.process.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            with job.lock:
                if line.startswith("EVENT "):
                    job.events.append(line[len("EVENT "):])
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    job.events.append(f"[stdout] {line}")
                    continue
                if {"epoch", "loss", "grad_norm", "lr"} <= entry.keys():
                    job.losses.append(entry["loss"])
                    job.grad_norms.append(entry["grad_norm"])
                    job.lr = entry["lr"]
                    if "val_loss" in entry:
                        job.val_losses.append(entry["val_loss"])
                        job.val_accs.append(entry["val_acc"])

    def get_status(self, name: str) -> TrainJobStatus:
        job = self._jobs[name]
        with job.lock:
            gpu_hours_used = job.gpu_hours_budget * len(job.losses) / max(1, job.total_epochs)
            return TrainJobStatus(
                name=name, epoch=len(job.losses), total_epochs=job.total_epochs,
                loss_history=list(job.losses), grad_norm_history=list(job.grad_norms),
                lr=job.lr, gpu_hours_used=gpu_hours_used, gpu_hours_budget=job.gpu_hours_budget,
                phase=job.phase(),
                val_loss_history=list(job.val_losses), val_acc_history=list(job.val_accs),
            )

    def get_new_events(self, name: str) -> list[str]:
        """Events (real process lifecycle/log lines) emitted since the last
        call - what dashboard/backend/live.py streams to the live log
        panel. Not idempotent by design: each event is returned once."""
        job = self._jobs[name]
        with job.lock:
            new = job.events[job._events_read_index:]
            job._events_read_index = len(job.events)
            return new

    def patch_lr(self, name: str, new_lr: float) -> None:
        job = self._jobs[name]
        job.lr_override_path.write_text(json.dumps({"lr": new_lr}))

    def stop(self, name: str) -> None:
        job = self._jobs[name]
        if job.process.poll() is None:
            job.process.terminate()
            try:
                job.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                job.process.kill()

    def step(self, name: str) -> bool:
        return self.get_status(name).phase == "Running"
