"""Small CPU-friendly image-classification training run (Fashion-MNIST by default).

The live dashboard always runs the default workload (Fashion-MNIST, small
CNN, Adam). The --dataset/--arch/--optimizer options exist so the research
experiments (experiments/collect_curves.py) can record learning curves of
other real workloads with the exact same instrumentation: MNIST, CIFAR-10
and a multilayer perceptron trained with SGD.

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

Requires the real dataset to be downloadable (internet access to
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
import time


def log_metric(epoch: int, loss: float, grad_norm: float, lr: float,
               val_loss: float | None = None, val_acc: float | None = None,
               epoch_seconds: float | None = None) -> None:
    entry = {"epoch": epoch, "loss": loss, "grad_norm": grad_norm, "lr": lr}
    if val_loss is not None:
        entry["val_loss"] = val_loss
        entry["val_acc"] = val_acc
    if epoch_seconds is not None:
        entry["epoch_seconds"] = epoch_seconds
    print(json.dumps(entry), flush=True)


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


CIFAR10_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def _cifar10_tensors(data_root: str, split: str):
    """Real CIFAR-10 (fast.ai mirror of the original images), decoded once
    and cached as a uint8 tensor file. Images in the mirror are grouped by
    class, so callers draw subsets through a fixed permutation."""
    import tarfile
    import urllib.request
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image

    root = Path(data_root) / "cifar10"
    cache = root / f"{split}.pt"
    if cache.exists():
        return torch.load(cache)
    archive = Path(data_root) / "cifar10.tgz"
    if not (root / split).exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            log_event(f"Downloading CIFAR-10 from {CIFAR10_URL}")
            urllib.request.urlretrieve(CIFAR10_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(Path(data_root))
    images, labels = [], []
    for label, name in enumerate(CIFAR10_CLASSES):
        for f in sorted((root / split / name).glob("*.png")):
            images.append(np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8))
            labels.append(label)
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(labels, dtype=torch.long)
    torch.save((x, y), cache)
    return x, y


def _fixed_subset(x, y, size: int):
    """A class-mixed subset that is identical for every run (fixed seed 0),
    independent of the run's own --seed."""
    import torch

    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))[:size]
    return x[order], y[order]


def _load_split(dataset: str, train: bool, size: int, data_root: str):
    """Real images from the official train (train=True) or test split."""
    import torch
    from torch.utils.data import Subset, TensorDataset
    from torchvision import datasets, transforms

    if dataset == "cifar10":
        x, y = _cifar10_tensors(data_root, "train" if train else "test")
        x, y = _fixed_subset(x, y, min(size, len(y)))
        return TensorDataset(x.float().div(255.0), y)
    cls = {"fashion_mnist": datasets.FashionMNIST, "mnist": datasets.MNIST}[dataset]
    full = cls(root=data_root, train=train, download=True, transform=transforms.ToTensor())
    return Subset(full, range(min(size, len(full))))


def _load_dataset(subset_size: int, data_root: str, dataset: str = "fashion_mnist"):
    """Real training images, downloaded if not already cached at data_root.
    No synthetic fallback: if this fails, it raises and main() logs a
    loud, clear error before letting the process exit non-zero - per
    project policy, a run is either real data or it doesn't run."""
    subset = _load_split(dataset, True, subset_size, data_root)
    log_event(f"Loaded real {dataset} subset ({len(subset)} images) from {data_root}")
    return subset


def _load_val_dataset(val_size: int, data_root: str, dataset: str = "fashion_mnist"):
    """Held-out validation images from the dataset's official test split,
    so the monitored loss is never computed on training images."""
    return _load_split(dataset, False, val_size, data_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subset-size", type=int, default=6000,
                         help="Keep this small - the point is a real but fast CPU run.")
    parser.add_argument("--data-root", default="/tmp/data",
                         help="Where datasets are cached/downloaded to.")
    parser.add_argument("--workers", type=int, default=2,
                         help="DataLoader worker processes. 0 for background mode "
                              "(avoid extra processes competing for CPU), higher for turbo.")
    parser.add_argument("--amp", action="store_true",
                         help="Mixed-precision training (turbo mode) - faster on a CUDA GPU, "
                              "no effect on CPU.")
    parser.add_argument("--cpu-threads", type=int, default=0,
                         help="Cap torch's intra-op thread count. 0 = torch's default (all "
                              "cores). Background mode passes a small number so a single "
                              "training process can't saturate every core even when nothing "
                              "else on the machine is competing for them - OS priority alone "
                              "doesn't help on an otherwise-idle machine.")
    parser.add_argument("--throttle-ms", type=int, default=0,
                         help="Sleep this many milliseconds after every batch. 0 = no "
                              "throttle. Background mode uses a small value to trade some "
                              "throughput for a visibly low, predictable CPU/GPU load.")
    parser.add_argument("--val-size", type=int, default=2000,
                         help="Images from the official test split used for per-epoch "
                              "validation loss/accuracy. 0 disables validation.")
    parser.add_argument("--dataset", choices=["fashion_mnist", "mnist", "cifar10"],
                        default="fashion_mnist")
    parser.add_argument("--arch", choices=["cnn", "mlp"], default="cnn",
                        help="cnn: two-conv CNN (three-conv for CIFAR-10); mlp: 256-128 MLP.")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam",
                        help="sgd uses momentum 0.9.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed torch's RNG (weights init + shuffling) for reproducible runs.")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch not installed - `pip install -r training/requirements.txt`.", file=sys.stderr)
        raise

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        log_event(f"Capped torch CPU threads to {args.cpu_threads} (background mode)")
    if args.seed is not None:
        torch.manual_seed(args.seed)

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

    class CifarCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = F.max_pool2d(F.relu(self.conv3(x)), 2)
            x = F.relu(self.fc1(x.flatten(1)))
            return self.fc2(x)

    class MLP(nn.Module):
        def __init__(self, in_features: int):
            super().__init__()
            self.fc1 = nn.Linear(in_features, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x.flatten(1)))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    def build_model():
        if args.arch == "mlp":
            return MLP(3 * 32 * 32 if args.dataset == "cifar10" else 28 * 28)
        return CifarCNN() if args.dataset == "cifar10" else SmallCNN()

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
        dataset = _load_dataset(args.subset_size, args.data_root, args.dataset)
        val_dataset = (_load_val_dataset(args.val_size, args.data_root, args.dataset)
                       if args.val_size > 0 else None)
    except Exception as exc:
        log_event(
            f"FATAL: could not load real {args.dataset} ({exc.__class__.__name__}: {exc}). "
            f"No synthetic fallback by design - check network access to download it, or "
            f"pre-populate --data-root ({args.data_root}) with the dataset."
        )
        raise

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=512) if val_dataset is not None else None

    def evaluate():
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                loss_sum += F.cross_entropy(out, labels, reduction="sum").item()
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.numel()
        model.train()
        return loss_sum / total, correct / total

    model = build_model().to(device)
    lr = args.lr
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.amp and not use_amp:
        log_event("--amp requested but no CUDA device available - training at normal precision")

    log_event(f"Starting training: {args.dataset}/{args.arch}/{args.optimizer}, {args.epochs} epochs, batch_size={args.batch_size}, "
              f"initial_lr={lr}, workers={args.workers}, amp={use_amp}, "
              f"cpu_threads={args.cpu_threads or 'default'}, throttle_ms={args.throttle_ms}")

    for epoch in range(1, args.epochs + 1):
        new_lr = read_lr_override(lr)
        if new_lr != lr:
            log_event(f"Learning rate patched: {lr:.6g} -> {new_lr:.6g}")
        lr = new_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        epoch_start = time.perf_counter()
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
            # Global L2 norm of the true (unscaled) gradient - under AMP the
            # raw .grad values are multiplied by the loss scale.
            scaler.unscale_(optimizer)
            grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters()
                                       if p.grad is not None)).item()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_grad_norm += grad_norm
            n_batches += 1

            if args.throttle_ms > 0:
                time.sleep(args.throttle_ms / 1000)

        val_loss, val_acc = evaluate() if val_loader is not None else (None, None)
        log_metric(epoch, total_loss / n_batches, total_grad_norm / n_batches, lr,
                   val_loss=val_loss, val_acc=val_acc,
                   epoch_seconds=time.perf_counter() - epoch_start)

    log_event("Training complete")


if __name__ == "__main__":
    main()
