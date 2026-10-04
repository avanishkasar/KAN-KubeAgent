"""Live session orchestration: runs a real local training subprocess,
samples real hardware telemetry, drives the real agent+KAN pipeline
against it as epochs complete, and broadcasts everything over WebSocket.

Single global session by design - this is a one-operator research
dashboard, not a multi-tenant service. Starting a new live run replaces
any previous one (the previous process is stopped first).

WebSocket message types: snapshot, hardware, metric, event, agent_step
(one per LangGraph node as it actually completes), decision (with the
KAN trace from kan_gate/introspect.py), done.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import WebSocket

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agents.graph import build_graph  # noqa: E402
from agents.local_process_client import LocalProcessTrainJobClient  # noqa: E402
from dashboard.backend import hardware  # noqa: E402
from kan_gate.gate import DEFAULT_CKPT_PATH, KANGate  # noqa: E402
from kan_gate.introspect import explain as explain_decision  # noqa: E402
from kan_gate.real_run_logger import save_decision_points  # noqa: E402

HARDWARE_SAMPLE_INTERVAL_S = 2.0
POLL_INTERVAL_S = 0.5
AGENT_NODES = ("metrics_watcher", "loss_curve_and_cost", "supervisor", "kan_gate", "execute")


def _summarize_step(node: str, state: dict) -> dict:
    """What each agent node actually produced, for the pipeline view."""
    if node == "metrics_watcher":
        detail = {"epochs_observed": len(state.get("loss_history", [])), "lr": state.get("lr")}
    elif node == "loss_curve_and_cost":
        detail = {"features": state.get("features")}
    elif node == "supervisor":
        detail = {"proposal": state.get("proposed_action"),
                  "rationale": state.get("proposal_rationale")}
    elif node == "kan_gate":
        detail = {"decision": state.get("gate_decision"), "score": state.get("gate_score")}
    else:
        log = state.get("audit_log") or [{}]
        detail = {"outcome": log[-1].get("outcome")}
    return {"node": node, "ts": time.time(), "detail": detail}


def _empty_snapshot(phase: str, job_name: str | None) -> dict[str, Any]:
    return {
        "phase": phase, "job_name": job_name, "current_job": job_name,
        "loss_history": [], "grad_norm_history": [], "val_loss_history": [], "val_acc_history": [],
        "decisions": [], "events": [], "hardware_history": [], "last_hardware": None,
        "agent_steps": [], "auto_run_count": 0, "total_epochs_trained": 0,
    }


class LiveSession:
    def __init__(self):
        self.client: LocalProcessTrainJobClient | None = None
        self.job_name: str | None = None
        self.current_job: str | None = None  # the subprocess running now (differs in auto mode)
        self.check_every: int = 3
        self.gate: KANGate | None = None
        self.graph = None
        self.agent_state: dict[str, Any] = {}
        self.connections: list[WebSocket] = []
        self.task: asyncio.Task | None = None
        self.running = False
        self.auto = False
        self.mode = "normal"
        self.gpu_index: int | None = None
        self._run_epochs = 30
        self._run_lr = 2e-4
        self._run_batch_size = 128
        self._run_subset_size = 6000
        self._run_seed = 0
        self._run_decisions_start = 0

        # Rolling snapshot so a client that connects mid-run (or
        # reconnects) can catch up instead of seeing a blank page.
        self.snapshot: dict[str, Any] = _empty_snapshot("idle", None)

    async def register(self, ws: WebSocket) -> None:
        await ws.accept()
        self.connections.append(ws)
        await ws.send_json({"type": "snapshot", "data": self.snapshot})

    def unregister(self, ws: WebSocket) -> None:
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)

    async def emit_event(self, message: str, level: str = "info") -> None:
        entry = {"ts": time.time(), "message": message, "level": level}
        self.snapshot["events"].append(entry)
        await self.broadcast({"type": "event", "data": entry})

    def load_gate(self) -> KANGate:
        try:
            return KANGate.load(DEFAULT_CKPT_PATH)
        except Exception:
            return KANGate()

    async def start(self, job_name: str, epochs: int, check_every: int, lr: float,
                    batch_size: int, subset_size: int, mode: str = "normal",
                    auto: bool = False, gpu_index: int | None = None) -> None:
        if self.running:
            await self.stop()

        self.client = LocalProcessTrainJobClient()
        self.job_name = job_name
        self.current_job = job_name
        self.check_every = check_every
        self.mode = mode
        self.auto = auto
        self.gpu_index = gpu_index
        self._run_epochs = epochs
        self._run_lr = lr
        self._run_batch_size = batch_size
        self._run_subset_size = subset_size
        self._run_seed = 0
        self._run_decisions_start = 0
        self.gate = self.load_gate()
        self.graph = build_graph(self.client, self.gate)
        self.agent_state = {"trainjob_name": job_name, "audit_log": []}
        self.snapshot = _empty_snapshot("starting", job_name)
        self.running = True

        await self.broadcast({"type": "snapshot", "data": self.snapshot})

        hw = await asyncio.to_thread(hardware.sample)
        if hw["gpu_available"]:
            gpu_names = ", ".join(f"[{g['index']}] {g['name']}" for g in hw["gpus"])
            gpu_report = f"{len(hw['gpus'])} GPU(s) detected: {gpu_names}"
            if gpu_index is not None and gpu_index < 0:
                gpu_report += ". CPU-only training was selected"
            elif gpu_index is not None:
                gpu_report += f" - training pinned to GPU {gpu_index}"
            elif len(hw["gpus"]) > 1:
                gpu_report += " - no GPU selected, CUDA will pick its own default"
        else:
            gpu_report = "No GPU detected (nvidia-smi not found) - training will run on CPU"
        await self.emit_event(
            f"Hardware check: {hw['cpu_count']} CPU cores, {gpu_report}. "
            f"Mode: {mode}{' (auto-continuous)' if auto else ''}"
        )

        await self.emit_event(
            f"Launching real training subprocess for '{job_name}' "
            f"({epochs} epochs, batch_size={batch_size}, subset_size={subset_size}, lr={lr})"
        )
        await asyncio.to_thread(
            self.client.create, job_name, epochs=epochs, lr=lr,
            batch_size=batch_size, subset_size=subset_size, mode=mode, gpu_index=gpu_index,
        )
        self.task = asyncio.create_task(self._run_loop())

    async def _harvest(self, job_name: str, phase: str) -> None:
        """Save this run's decisions plus its full monitored curve, so the
        gate can later be retrained on hindsight labels (kan_gate/train.py)."""
        run_decisions = self.snapshot["decisions"][self._run_decisions_start:]
        status = await asyncio.to_thread(self.client.get_status, job_name)
        monitored = status.to_history().loss_history
        saved_path = await asyncio.to_thread(
            save_decision_points, job_name, run_decisions,
            monitored_loss=list(monitored), grad_norms=list(status.grad_norm_history), phase=phase,
        )
        if saved_path:
            usable = "usable for hindsight retraining" if phase == "Completed" else \
                "kept for the record (stopped early, so no observed future to label from)"
            await self.emit_event(
                f"Saved run '{job_name}' ({len(run_decisions)} decision points, "
                f"{len(monitored)} epochs) to {saved_path} - {usable}"
            )

    async def stop(self) -> None:
        self.auto = False
        job = self.current_job
        if self.client and job:
            await self.emit_event(f"Stopping '{job}' (user requested)", level="warning")
            await asyncio.to_thread(self.client.stop, job)
        self.running = False
        if self.task:
            self.task.cancel()
        # _run_loop's own "phase != Running" branch sends this when the
        # process ends on its own; a user-initiated stop cancels that loop
        # before it gets there, so send it here instead - otherwise the
        # frontend never learns the run ended and Start/Stop stay stuck.
        self.snapshot["phase"] = "Stopped"
        if self.client and job:
            await self._harvest(job, "Stopped")
        await self.broadcast({"type": "done", "data": {"phase": "Stopped"}})

    def _run_graph(self, state: dict, loop: asyncio.AbstractEventLoop) -> dict:
        """Runs the agent graph node by node (LangGraph streaming), pushing
        an agent_step event to the browser as each node really finishes."""
        final = state
        for update in self.graph.stream(state, stream_mode="updates"):
            for node, node_state in update.items():
                final = node_state
                step = _summarize_step(node, node_state)
                self.snapshot["agent_steps"] = (self.snapshot["agent_steps"] + [step])[-50:]
                asyncio.run_coroutine_threadsafe(
                    self.broadcast({"type": "agent_step", "data": step}), loop)
        return final

    async def _run_loop(self) -> None:
        job_name = self.current_job
        last_hardware_sample = 0.0
        last_epoch_seen = 0
        last_gate_epoch = 0
        loop = asyncio.get_running_loop()

        try:
            while self.running:
                now = time.monotonic()

                if now - last_hardware_sample >= HARDWARE_SAMPLE_INTERVAL_S:
                    sample = await asyncio.to_thread(hardware.sample)
                    last_hardware_sample = now
                    self.snapshot["last_hardware"] = sample
                    self.snapshot["hardware_history"].append(sample)
                    self.snapshot["hardware_history"] = self.snapshot["hardware_history"][-300:]
                    await self.broadcast({"type": "hardware", "data": sample})

                status = await asyncio.to_thread(self.client.get_status, job_name)
                self.snapshot["phase"] = status.phase

                if status.epoch > last_epoch_seen:
                    for i in range(last_epoch_seen, status.epoch):
                        metric = {
                            "epoch": i + 1, "loss": status.loss_history[i],
                            "grad_norm": status.grad_norm_history[i], "lr": status.lr,
                        }
                        if i < len(status.val_loss_history):
                            metric["val_loss"] = status.val_loss_history[i]
                            metric["val_acc"] = status.val_acc_history[i]
                            self.snapshot["val_loss_history"].append(metric["val_loss"])
                            self.snapshot["val_acc_history"].append(metric["val_acc"])
                        self.snapshot["loss_history"].append(metric["loss"])
                        self.snapshot["grad_norm_history"].append(metric["grad_norm"])
                        self.snapshot["total_epochs_trained"] += 1
                        await self.broadcast({"type": "metric", "data": metric})
                    last_epoch_seen = status.epoch

                for line in await asyncio.to_thread(self.client.get_new_events, job_name):
                    await self.emit_event(line, level="process")

                if status.epoch >= last_gate_epoch + self.check_every and status.epoch > 0:
                    last_gate_epoch = status.epoch
                    await self.emit_event(
                        f"Epoch {status.epoch}: invoking agent loop (Metrics Watcher -> "
                        f"Loss-Curve Analyst/Cost Estimator -> Supervisor -> KAN gate)"
                    )
                    self.agent_state = await asyncio.to_thread(self._run_graph, self.agent_state, loop)
                    entry = self.agent_state["audit_log"][-1]
                    entry["explanation"] = await asyncio.to_thread(
                        explain_decision, self.gate, entry["features"])
                    entry["run"] = job_name
                    self.snapshot["decisions"].append(entry)
                    await self.broadcast({"type": "decision", "data": entry})

                    level = {"continue": "info", "adjust_lr": "warning", "early_stop": "critical"}[entry["decision"]]
                    top = max(entry["explanation"]["contributions"].items(), key=lambda kv: abs(kv[1]))
                    await self.emit_event(
                        f"KAN gate decision: {entry['decision']} (score={entry['gate_score']:.1f}, "
                        f"largest contribution: {top[0]} {top[1]:+.1f}) -> {entry['outcome']}",
                        level=level,
                    )
                    if entry["decision"] == "adjust_lr":
                        await self.emit_event(f"Patched learning rate to {entry.get('new_lr', '?')}")
                    elif entry["decision"] == "early_stop":
                        await self.emit_event(
                            f"Terminating training subprocess for '{job_name}' (KAN gate authority)",
                            level="critical",
                        )

                if status.phase != "Running":
                    await self.emit_event(f"Training ended: {status.phase}",
                                          level="info" if status.phase == "Completed" else "warning")
                    await self._harvest(job_name, status.phase)

                    if not self.auto or status.phase != "Completed":
                        self.running = False
                        await self.broadcast({"type": "done", "data": {"phase": status.phase}})
                        break

                    # Auto-continuous: this run finished cleanly and the
                    # operator asked for continuous background training, so
                    # start a fresh real run (new seed/job name) immediately
                    # instead of stopping. The chart resets per run (nothing
                    # carries over model weights between runs), while
                    # auto_run_count/total_epochs_trained accumulate.
                    self._run_seed += 1
                    self.snapshot["auto_run_count"] += 1
                    job_name = f"{self.job_name}-auto{self._run_seed}"
                    self.current_job = job_name
                    self.snapshot["current_job"] = job_name
                    self._run_decisions_start = len(self.snapshot["decisions"])
                    for key in ("loss_history", "grad_norm_history", "val_loss_history", "val_acc_history"):
                        self.snapshot[key] = []
                    self.snapshot["phase"] = "starting"
                    self.agent_state = {"trainjob_name": job_name, "audit_log": []}
                    await self.broadcast({"type": "snapshot", "data": self.snapshot})
                    await self.emit_event(
                        f"Auto-continuous mode: starting real run #{self.snapshot['auto_run_count'] + 1} "
                        f"'{job_name}' ({self._run_epochs} epochs, mode={self.mode}) - "
                        f"{self.snapshot['total_epochs_trained']} epochs trained so far this session"
                    )
                    await asyncio.to_thread(
                        self.client.create, job_name, epochs=self._run_epochs, lr=self._run_lr,
                        batch_size=self._run_batch_size, subset_size=self._run_subset_size,
                        mode=self.mode, gpu_index=self.gpu_index,
                    )
                    last_epoch_seen = 0
                    last_gate_epoch = 0
                    continue

                await asyncio.sleep(POLL_INTERVAL_S)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            await self.emit_event(f"Live loop error: {exc}", level="critical")
            self.running = False


session = LiveSession()
