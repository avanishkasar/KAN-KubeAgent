"""Real TrainJobClient backed by a live Kubeflow `TrainJob` on Kubernetes.

Implements the same interface as agents/trainjob_client.py's
MockTrainJobClient, so agents/graph.py does not change when swapping this
in - see k8s/README.md for cluster setup.

NOT exercised against a live cluster from the Claude Code sandbox this
repo was built in - that sandbox's network policy blocks registry.k8s.io,
so Minikube cannot start there (see k8s/README.md). Written against the
Kubernetes Python client's documented API and training/fashion_mnist_cnn.py's
METRIC log-line contract; validate against your own cluster and expect to
need small fixes if the installed Trainer version's API differs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from agents.trainjob_client import TrainJobStatus


@dataclass
class KubeflowTrainJobClient:
    namespace: str = "default"
    gpu_hours_budget: float = 10.0
    total_epochs: int = 30

    def __post_init__(self):
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        self._core_v1 = client.CoreV1Api()
        self._custom = client.CustomObjectsApi()
        self._group, self._version, self._plural = "trainer.kubeflow.org", "v1alpha1", "trainjobs"

    def _pod_name(self, trainjob_name: str) -> str:
        pods = self._core_v1.list_namespaced_pod(
            self.namespace, label_selector=f"trainer.kubeflow.org/trainjob-name={trainjob_name}"
        )
        if not pods.items:
            raise RuntimeError(f"No pod found for TrainJob '{trainjob_name}' yet")
        return pods.items[0].metadata.name

    def _parse_metrics(self, trainjob_name: str) -> tuple[list[float], list[float], float]:
        pod_name = self._pod_name(trainjob_name)
        logs = self._core_v1.read_namespaced_pod_log(pod_name, self.namespace)
        losses, grads, lr = [], [], 2e-4
        for line in logs.splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if {"epoch", "loss", "grad_norm", "lr"} <= entry.keys():
                losses.append(entry["loss"])
                grads.append(entry["grad_norm"])
                lr = entry["lr"]
        return losses, grads, lr

    def get_status(self, name: str) -> TrainJobStatus:
        losses, grads, lr = self._parse_metrics(name)
        trainjob = self._custom.get_namespaced_custom_object(
            self._group, self._version, self.namespace, self._plural, name
        )
        phase = trainjob.get("status", {}).get("phase", "Running")
        gpu_hours_used = self.gpu_hours_budget * len(losses) / max(1, self.total_epochs)

        return TrainJobStatus(
            name=name, epoch=len(losses), total_epochs=self.total_epochs,
            loss_history=losses, grad_norm_history=grads, lr=lr,
            gpu_hours_used=gpu_hours_used, gpu_hours_budget=self.gpu_hours_budget,
            phase=phase,
        )

    def patch_lr(self, name: str, new_lr: float) -> None:
        """Writes a ConfigMap that training/fashion_mnist_cnn.py polls each
        epoch - avoids requiring a training-framework-specific live-patch
        API, at the cost of a one-epoch delay before it takes effect."""
        from kubernetes import client
        from kubernetes.client.rest import ApiException

        cm_name = f"{name}-lr-override"
        body = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(name=cm_name),
            data={"lr": str(new_lr)},
        )
        try:
            self._core_v1.replace_namespaced_config_map(cm_name, self.namespace, body)
        except ApiException as exc:
            if exc.status == 404:
                self._core_v1.create_namespaced_config_map(self.namespace, body)
            else:
                raise

    def stop(self, name: str) -> None:
        self._custom.delete_namespaced_custom_object(
            self._group, self._version, self.namespace, self._plural, name
        )

    def step(self, name: str) -> bool:
        """No-op for the real client - training progresses on its own
        inside the pod; this method exists only so MockTrainJobClient and
        KubeflowTrainJobClient satisfy the same TrainJobClient protocol
        for callers like agents/run_loop.py that drive a mock job epoch by
        epoch. Against a real cluster, poll get_status() on a timer
        instead (see agents/run_loop.py's mock-only driving loop vs. a
        production polling loop you'd add here)."""
        return self.get_status(name).phase == "Running"
