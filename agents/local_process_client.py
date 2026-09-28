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

from agents.trainjob_client import TrainJobStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = REPO_ROOT / "training" / "fashion_mnist_cnn.py"


@dataclass
class _RunningJob:
    process: subprocess.Popen
    lr_override_path: Path
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
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
               gpu_hours_budget: float = 10.0) -> TrainJobStatus:
        lr_override_path = Path(tempfile.gettempdir()) / f"kan-kubeagent-{name}-lr.json"
        lr_override_path.write_text(json.dumps({"lr": lr}))

        env = {**os.environ, "LR_OVERRIDE_FILE": str(lr_override_path), "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen(
            [sys.executable, str(TRAINING_SCRIPT),
             "--epochs", str(epochs), "--lr", str(lr),
             "--batch-size", str(batch_size), "--subset-size", str(subset_size)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            cwd=str(REPO_ROOT), env=env,
        )
        job = _RunningJob(process=process, lr_override_path=lr_override_path,
                           lr=lr, total_epochs=epochs, gpu_hours_budget=gpu_hours_budget)
        self._jobs[name] = job

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

    def get_status(self, name: str) -> TrainJobStatus:
        job = self._jobs[name]
        with job.lock:
            gpu_hours_used = job.gpu_hours_budget * len(job.losses) / max(1, job.total_epochs)
            return TrainJobStatus(
                name=name, epoch=len(job.losses), total_epochs=job.total_epochs,
                loss_history=list(job.losses), grad_norm_history=list(job.grad_norms),
                lr=job.lr, gpu_hours_used=gpu_hours_used, gpu_hours_budget=job.gpu_hours_budget,
                phase=job.phase(),
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
