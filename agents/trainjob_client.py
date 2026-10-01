"""TrainJob client interface.

Agents never touch the Kubernetes API directly - they go through this
client, which has two implementations:

- MockTrainJobClient: an in-memory simulated job, used for developing and
  testing the agent loop before a real cluster is wired up.
- KubeflowTrainJobClient (agents/kubeflow_client.py, added once Minikube +
  Kubeflow Trainer are installed - see research/proposal/methodology_draft.md
  Section 2.1): wraps the real `kubeflow.trainer.TrainerClient`.

Both implement the same interface so the agent graph in agents/graph.py
does not change when swapping from mock to real.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from kan_gate.features import TrainingHistory


@dataclass
class TrainJobStatus:
    name: str
    epoch: int
    total_epochs: int
    loss_history: list[float]
    grad_norm_history: list[float]
    lr: float
    gpu_hours_used: float
    gpu_hours_budget: float
    phase: str = "Running"  # Running | Stopped | Completed
    val_loss_history: list[float] = field(default_factory=list)
    val_acc_history: list[float] = field(default_factory=list)

    def to_history(self, window: int = 5) -> TrainingHistory:
        # Monitor held-out loss when the job reports it - early stopping on
        # training loss can't see overfitting.
        monitored = (self.val_loss_history
                     if len(self.val_loss_history) == len(self.loss_history) and self.val_loss_history
                     else self.loss_history)
        return TrainingHistory(
            loss_history=monitored,
            grad_norm_history=self.grad_norm_history,
            lr=self.lr,
            gpu_hours_used=self.gpu_hours_used,
            gpu_hours_budget=self.gpu_hours_budget,
            window=window,
        )


class TrainJobClient(Protocol):
    def get_status(self, name: str) -> TrainJobStatus: ...
    def patch_lr(self, name: str, new_lr: float) -> None: ...
    def stop(self, name: str) -> None: ...
    def step(self, name: str) -> bool:
        """Advance the job by one epoch. Returns False once the job has
        stopped or completed."""
        ...


@dataclass
class MockTrainJobClient:
    """In-memory TrainJob for developing/testing the agent loop without a
    real cluster. Backed by the same synthetic-curve generator as
    kan_gate/synthetic.py, one epoch revealed at a time."""

    jobs: dict[str, TrainJobStatus] = field(default_factory=dict)
    _full_curves: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)

    def create(self, name: str, total_epochs: int = 30, plateau_at_epoch: int | None = None,
               gpu_hours_budget: float = 10.0, seed: int | None = None) -> TrainJobStatus:
        from kan_gate.synthetic import generate_synthetic_history

        full = generate_synthetic_history(
            num_epochs=total_epochs,
            plateau_at_epoch=plateau_at_epoch,
            gpu_hours_budget=gpu_hours_budget,
            seed=seed,
        )
        self._full_curves[name] = (full.loss_history, full.grad_norm_history)
        status = TrainJobStatus(
            name=name, epoch=0, total_epochs=total_epochs,
            loss_history=[], grad_norm_history=[], lr=2e-4,
            gpu_hours_used=0.0, gpu_hours_budget=gpu_hours_budget,
        )
        self.jobs[name] = status
        return status

    def get_status(self, name: str) -> TrainJobStatus:
        return self.jobs[name]

    def patch_lr(self, name: str, new_lr: float) -> None:
        self.jobs[name].lr = new_lr

    def stop(self, name: str) -> None:
        self.jobs[name].phase = "Stopped"

    def step(self, name: str) -> bool:
        job = self.jobs[name]
        if job.phase != "Running":
            return False
        losses, grads = self._full_curves[name]
        if job.epoch >= job.total_epochs:
            job.phase = "Completed"
            return False
        job.loss_history.append(losses[job.epoch])
        job.grad_norm_history.append(grads[job.epoch])
        job.epoch += 1
        job.gpu_hours_used += job.gpu_hours_budget / job.total_epochs
        if job.epoch >= job.total_epochs:
            job.phase = "Completed"
            return False
        return True
