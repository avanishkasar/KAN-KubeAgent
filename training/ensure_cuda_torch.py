"""Installs the CUDA build of PyTorch when this machine has an NVIDIA GPU but
the installed torch is the CPU-only wheel (what a plain `pip install torch`
gives on Windows and most Linux setups). Run by start.ps1 and start.sh; safe
to run any time, it does nothing when CUDA already works or no NVIDIA GPU
is present.

    python training/ensure_cuda_torch.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys

INDEX = "https://download.pytorch.org/whl/{tag}"
# (minimum driver CUDA version, wheel tag), newest first
TAGS = [((12, 8), "cu128"), ((12, 6), "cu126"), ((12, 4), "cu124"), ((12, 1), "cu121")]


def driver_cuda_version(smi_output: str) -> tuple[int, int] | None:
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi_output)
    return (int(m.group(1)), int(m.group(2))) if m else None


def pick_tag(version: tuple[int, int]) -> str | None:
    for minimum, tag in TAGS:
        if version >= minimum:
            return tag
    return None


def torch_has_cuda() -> bool:
    out = subprocess.run([sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                         capture_output=True)
    return out.returncode == 0


def main() -> int:
    smi = shutil.which("nvidia-smi")
    if not smi:
        print("No NVIDIA GPU detected (nvidia-smi not found); training will use the CPU.")
        return 0
    if torch_has_cuda():
        print("PyTorch already has CUDA support; training can use the GPU.")
        return 0

    text = subprocess.run([smi], capture_output=True, text=True).stdout
    version = driver_cuda_version(text)
    tag = pick_tag(version) if version else None
    if not tag:
        print(f"Found an NVIDIA GPU but could not choose a CUDA build of PyTorch (driver CUDA version: {version}). "
              "Update the NVIDIA driver, or install PyTorch manually from https://pytorch.org/get-started/locally/")
        return 0

    print(f"NVIDIA GPU found, driver supports CUDA {version[0]}.{version[1]}. Installing the {tag} build of PyTorch (about 2.5 GB)...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision"], check=False)
    result = subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision",
                             "--index-url", INDEX.format(tag), "--extra-index-url", "https://pypi.org/simple"])
    if result.returncode != 0 or not torch_has_cuda():
        print("The CUDA install did not work. Training will fall back to the CPU. "
              "See training/requirements.txt for the manual command.")
        return 0
    print("CUDA PyTorch installed. Training will now use the GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
