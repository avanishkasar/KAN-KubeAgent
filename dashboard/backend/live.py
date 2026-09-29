"""Live session orchestration: runs a real local training subprocess,
samples real hardware telemetry, drives the real agent+KAN pipeline
against it as epochs complete, and broadcasts everything over WebSocket.

Single global session by design - this is a one-operator research
dashboard, not a multi-tenant service. Starting a new live run replaces
any previous one (the previous process is stopped first).
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
from kan_gate.real_run_logger import save_decision_points  # noqa: E402

HARDWARE_SAMPLE_INTERVAL_S = 2.0
POLL_INTERVAL_S = 0.5


class LiveSession:
    def __init__(self):
        self.client: LocalProcessTrainJobClient | None = None
        self.job_name: str | None = None
        self.check_every: int = 3
        self.gate: KANGate | None = None
        self.graph = None
        self.agent_state: dict[str, Any] = {}
        self.connections: list[WebSocket] = []
        self.task: asyncio.Task | None = None
        self.running = False

        # Rolling snapshot so a client that connects mid-run (or
        # reconnects) can catch up instead of seeing a blank page.
        self.snapshot: dict[str, Any] = {
            "phase": "idle", "job_name": None, "loss_history": [], "grad_norm_history": [],
            "decisions": [], "events": [], "hardware_history": [], "last_hardware": None,
        }

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
                     batch_size: int, subset_size: int) -> None:
        if self.running:
            await self.stop()

        self.client = LocalProcessTrainJobClient()
        self.job_name = job_name
        self.check_every = check_every
        self.gate = self.load_gate()
        self.graph = build_graph(self.client, self.gate)
        self.agent_state = {"trainjob_name": job_name, "audit_log": []}
        self.snapshot = {
            "phase": "starting", "job_name": job_name, "loss_history": [], "grad_norm_history": [],
            "decisions": [], "events": [], "hardware_history": [], "last_hardware": None,
        }
        self.running = True

        await self.broadcast({"type": "snapshot", "data": self.snapshot})
        await self.emit_event(
            f"Launching real training subprocess for '{job_name}' "
            f"({epochs} epochs, batch_size={batch_size}, subset_size={subset_size}, lr={lr})"
        )
        await asyncio.to_thread(
            self.client.create, job_name, epochs=epochs, lr=lr,
            batch_size=batch_size, subset_size=subset_size,
        )
        self.task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self.client and self.job_name:
            await self.emit_event(f"Stopping '{self.job_name}' (user requested)", level="warning")
            await asyncio.to_thread(self.client.stop, self.job_name)
        self.running = False
        if self.task:
            self.task.cancel()
        # _run_loop's own "phase != Running" branch sends this when the
        # process ends on its own; a user-initiated stop cancels that loop
        # before it gets there, so send it here instead - otherwise the
        # frontend never learns the run ended and Start/Stop stay stuck.
        self.snapshot["phase"] = "Stopped"
        if self.job_name:
            saved_path = await asyncio.to_thread(
                save_decision_points, self.job_name, self.snapshot["decisions"]
            )
            if saved_path:
                await self.emit_event(
                    f"Saved {len(self.snapshot['decisions'])} real decision points to "
                    f"{saved_path} for gate retraining (python -m kan_gate.train)"
                )
        await self.broadcast({"type": "done", "data": {"phase": "Stopped"}})

    async def _run_loop(self) -> None:
        job_name = self.job_name
        last_hardware_sample = 0.0
        last_epoch_seen = 0
        last_gate_epoch = 0

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
                        self.snapshot["loss_history"].append(metric["loss"])
                        self.snapshot["grad_norm_history"].append(metric["grad_norm"])
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
                    self.agent_state = await asyncio.to_thread(self.graph.invoke, self.agent_state)
                    entry = self.agent_state["audit_log"][-1]
                    self.snapshot["decisions"].append(entry)
                    await self.broadcast({"type": "decision", "data": entry})

                    level = {"continue": "info", "adjust_lr": "warning", "early_stop": "critical"}[entry["decision"]]
                    await self.emit_event(
                        f"KAN gate decision: {entry['decision']} (score={entry['gate_score']:.1f}) "
                        f"-> {entry['outcome']}", level=level,
                    )
                    if entry["decision"] == "adjust_lr":
                        await self.emit_event(f"Patched learning rate to {entry.get('new_lr', '?')}")
                    elif entry["decision"] == "early_stop":
                        await self.emit_event(
                            f"Terminating training subprocess for '{job_name}' (KAN gate authority)",
                            level="critical",
                        )

                if status.phase != "Running":
                    self.running = False
                    await self.emit_event(f"Training ended: {status.phase}",
                                           level="info" if status.phase == "Completed" else "warning")
                    saved_path = await asyncio.to_thread(
                        save_decision_points, job_name, self.snapshot["decisions"]
                    )
                    if saved_path:
                        await self.emit_event(
                            f"Saved {len(self.snapshot['decisions'])} real decision points to "
                            f"{saved_path} for gate retraining (python -m kan_gate.train)"
                        )
                    await self.broadcast({"type": "done", "data": {"phase": status.phase}})
                    break

                await asyncio.sleep(POLL_INTERVAL_S)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            await self.emit_event(f"Live loop error: {exc}", level="critical")
            self.running = False


session = LiveSession()
