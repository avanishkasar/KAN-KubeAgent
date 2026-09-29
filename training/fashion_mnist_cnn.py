"""Small CPU-friendly CNN fine-tune on Fashion-MNIST.

Runs as a real subprocess for local/live demos (agents/local_process_client.py)
or inside a TrainJob pod on a real cluster (k8s/trainjob-example.yaml,
agents/kubeflow_client.py). Prints one METRIC json line per epoch to
stdout - both clients read these back to build the loss/gradient history
the KAN gate scores. This keeps the real data path simple (no
Prometheus/metrics-server dependency) while staying a real, if small,
training run - per research/proposal/methodology_draft.md Section 6.1.

Also honours a live learning-rate patch, checked once per epoch:
- LR_OVERRIDE_FILE env var (local/live mode): a JSON file
  {"lr": <float>} that agents/local_process_client.py's patch_lr() writes.
- TRAINJOB_NAME env var (real cluster mode): a ConfigMap
  `{trainjob_name}-lr-override` that agents/kubeflow_client.py writes.

Requires real Fashion-MNIST to be downloadable (internet access to
pytorch's dataset mirror). There is no synthetic-data fallback here on
purpose - per project policy, this script either trains on real data or
fails loudly; it never silently substitutes fake data. If you're on a
machine without internet access, download the dataset ahead of time to
the `--data-root` directory (default /tmp/data) so torchvision finds it
already cached, or run against a network with access.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys


def log_metric(epoch: int, loss: float, grad_norm: float, lr: float) -> None:
    print(json.dumps({"epoch": epoch, "loss": loss, "grad_norm": grad_norm, "lr": lr}), flush=True)


def log_event(message: str) -> None:
    """A non-metric status line, prefixed so readers can tell it apart from
    METRIC json lines without parsing every line as JSON first."""
    print(f"EVENT {message}", flush=True)


def read_lr_override(default_lr: float) -> float:
    """Best-effort read of a live LR override written by the KAN-gated
    executor - a local JSON file in local/live mode, a ConfigMap on a real
    cluster. Falls back to the current LR if neither is set/reachable."""
    import os

    override_file = os.environ.get("LR_OVERRIDE_FILE")
    if override_file:
        try:
            with open(override_file) as f:
                return float(json.load(f)["lr"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
            return default_lr

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


def _load_dataset(subset_size: int, data_root: str):
    """Real Fashion-MNIST, downloaded if not already cached at data_root.
    No synthetic fallback: if this fails, it raises and main() logs a
    loud, clear error before letting the process exit non-zero - per
    project policy, a run is either real data or it doesn't run."""
    from torch.utils.data import Subset
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    subset = Subset(full_train, range(min(subset_size, len(full_train))))
    log_event(f"Loaded real Fashion-MNIST subset ({len(subset)} images) from {data_root}")
    return subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subset-size", type=int, default=6000,
                         help="Keep this small - the point is a real but fast CPU run.")
    parser.add_argument("--data-root", default="/tmp/data",
                         help="Where Fashion-MNIST is cached/downloaded to.")
    parser.add_argument("--workers", type=int, default=2,
                         help="DataLoader worker processes. 0 for background mode "
                              "(avoid extra processes competing for CPU), higher for turbo.")
    parser.add_argument("--amp", action="store_true",
                         help="Mixed-precision training (turbo mode) - faster on a CUDA GPU, "
                              "no effect on CPU.")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch not installed - `pip install -r training/requirements.txt`.", file=sys.stderr)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_event(f"Training device: {device}")
    if device.type == "cpu" and shutil.which("nvidia-smi") is not None:
        log_event(
            "NOTE: nvidia-smi found a GPU on this machine, but this PyTorch build has no "
            "CUDA support (torch.cuda.is_available() is False) - training will run on CPU. "
            "This is almost always a pip-installed CPU-only wheel, not missing hardware. "
            "Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121 "
            "(see training/requirements.txt)."
        )

    try:
        dataset = _load_dataset(args.subset_size, args.data_root)
    except Exception as exc:
        log_event(
            f"FATAL: could not load real Fashion-MNIST ({exc.__class__.__name__}: {exc}). "
            f"No synthetic fallback by design - check network access to download it, or "
            f"pre-populate --data-root ({args.data_root}) with the dataset."
        )
        raise

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, persistent_workers=args.workers > 0)

    model = SmallCNN().to(device)
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.amp and not use_amp:
        log_event("--amp requested but no CUDA device available - training at normal precision")

    log_event(f"Starting training: {args.epochs} epochs, batch_size={args.batch_size}, "
              f"initial_lr={lr}, workers={args.workers}, amp={use_amp}")

    for epoch in range(1, args.epochs + 1):
        new_lr = read_lr_override(lr)
        if new_lr != lr:
            log_event(f"Learning rate patched: {lr:.6g} -> {new_lr:.6g}")
        lr = new_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(images)
                loss = F.cross_entropy(output, labels)
            scaler.scale(loss).backward()
            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_grad_norm += grad_norm
            n_batches += 1

        log_metric(epoch, total_loss / n_batches, total_grad_norm / n_batches, lr)

    log_event("Training complete")


if __name__ == "__main__":
    main()
