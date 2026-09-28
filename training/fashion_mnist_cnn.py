"""Small CPU-friendly CNN fine-tune on Fashion-MNIST.

Runs inside the TrainJob pod (k8s/trainjob-example.yaml). Prints one
METRIC json line per epoch to stdout - agents/kubeflow_client.py reads
these back via `kubectl logs` to build the loss/gradient history the KAN
gate scores. This keeps the real data path simple (no Prometheus/metrics
server dependency) while staying a real, if small, training run - per
research/proposal/methodology_draft.md Section 6.1.

Also honours a live learning-rate patch: agents/executor.py's "adjust_lr"
action writes the new LR into a ConfigMap
(`{trainjob_name}-lr-override`); this script polls for it once per epoch.
"""
from __future__ import annotations

import argparse
import json
import sys


def log_metric(epoch: int, loss: float, grad_norm: float, lr: float) -> None:
    print(json.dumps({"epoch": epoch, "loss": loss, "grad_norm": grad_norm, "lr": lr}), flush=True)


def read_lr_override(default_lr: float) -> float:
    """Best-effort read of a live LR override written by the KAN-gated
    executor. Falls back to the current LR if none is set or the
    Kubernetes client isn't available (e.g. running outside a pod)."""
    import os

    trainjob_name = os.environ.get("TRAINJOB_NAME")
    namespace = os.environ.get("TRAINJOB_NAMESPACE", "default")
    if not trainjob_name:
        return default_lr
    try:
        from kubernetes import client, config

        config.load_incluster_config()
        v1 = client.CoreV1Api()
        cm = v1.read_namespaced_config_map(f"{trainjob_name}-lr-override", namespace)
        return float(cm.data["lr"])
    except Exception:
        return default_lr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subset-size", type=int, default=6000,
                         help="Keep this small - the point is a real but fast CPU run.")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
    except ImportError:
        print("torch/torchvision not installed - see k8s/trainjob-example.yaml "
              "for the pip install step, or `pip install torch torchvision` "
              "to run this locally.", file=sys.stderr)
        raise

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(32 * 7 * 7, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.FashionMNIST(root="/tmp/data", train=True, download=True, transform=transform)
    subset = Subset(full_train, range(min(args.subset_size, len(full_train))))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

    model = SmallCNN()
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, args.epochs + 1):
        lr = read_lr_override(lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0
        for images, labels in loader:
            optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            optimizer.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm
            n_batches += 1

        log_metric(epoch, total_loss / n_batches, total_grad_norm / n_batches, lr)


if __name__ == "__main__":
    main()
