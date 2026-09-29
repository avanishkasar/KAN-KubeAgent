#!/usr/bin/env bash
# One-command startup for Linux/DGX machines: creates the virtualenv if
# missing, installs dependencies, trains the KAN gate checkpoint if it
# isn't there yet, then launches the dashboard server detached (nohup) so
# it keeps running after this terminal closes - no browser required for
# training to continue, the browser is only a live viewer.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venv"
LOG_FILE="$REPO_ROOT/dashboard.log"
PID_FILE="$REPO_ROOT/dashboard.pid"
PORT="${PORT:-8000}"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Dashboard already running (pid $(cat "$PID_FILE")). Open http://localhost:${PORT}"
    echo "To restart: kill $(cat "$PID_FILE") && ./start.sh"
    exit 0
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing dependencies (this only downloads what's missing)..."
pip install --quiet --upgrade pip
pip install --quiet -r dashboard/backend/requirements.txt -r agents/requirements.txt -r training/requirements.txt

if command -v nvidia-smi >/dev/null 2>&1; then
    if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "NOTE: nvidia-smi found a GPU, but the installed torch has no CUDA support."
        echo "      Training will run on CPU until you run:"
        echo "        pip install torch --index-url https://download.pytorch.org/whl/cu121 && pip install torchvision"
        echo "      See training/requirements.txt for details."
    else
        echo "GPU detected and torch has CUDA support - training will use the GPU."
    fi
else
    echo "No NVIDIA GPU detected (no nvidia-smi) - training will run on CPU."
fi

if [ ! -f "kan_gate/checkpoints/kan_gate_config.yml" ]; then
    echo "No KAN gate checkpoint found - training one on synthetic data (fast, one-time)..."
    python3 -m kan_gate.train --synthetic --steps 250
fi

echo "Starting dashboard server in the background (survives this terminal closing)..."
nohup python3 -m uvicorn dashboard.backend.app:app --host 0.0.0.0 --port "$PORT" \
    > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
sleep 1

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo ""
    echo "Dashboard is running (pid $(cat "$PID_FILE")), logging to $LOG_FILE"
    echo "Open http://localhost:${PORT} in a browser any time - training keeps going without it open."
    echo "To stop it: kill \$(cat dashboard.pid)"
else
    echo "Server failed to start - check $LOG_FILE"
    exit 1
fi
