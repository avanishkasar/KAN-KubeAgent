# One-command startup for Windows: creates the virtualenv if missing,
# installs dependencies, trains the KAN gate checkpoint if it isn't there
# yet, then launches the dashboard server as a detached background process
# so it keeps running after this window closes - no browser required for
# training to continue, the browser is only a live viewer.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$VenvDir = Join-Path $RepoRoot ".venv"
$LogFile = Join-Path $RepoRoot "dashboard.log"
$PidFile = Join-Path $RepoRoot "dashboard.pid"
$Port = if ($env:PORT) { $env:PORT } else { "8000" }

if (Test-Path $PidFile) {
    $existingPid = Get-Content $PidFile
    if (Get-Process -Id $existingPid -ErrorAction SilentlyContinue) {
        Write-Host "Dashboard already running (pid $existingPid). Open http://localhost:$Port"
        Write-Host "To restart: Stop-Process -Id $existingPid; .\start.ps1"
        exit 0
    }
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtualenv at $VenvDir"
    python -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "Installing dependencies (this only downloads what's missing)..."
& $VenvPython -m pip install --quiet --upgrade pip
& $VenvPython -m pip install --quiet -r dashboard\backend\requirements.txt -r agents\requirements.txt -r training\requirements.txt -r kan_gate\requirements.txt

& $VenvPython training\ensure_cuda_torch.py

if (-not (Test-Path "kan_gate\checkpoints\kan_gate_config.yml")) {
    Write-Host "No KAN gate checkpoint found - training one on the bundled real learning curves (one-time)..."
    & $VenvPython -m kan_gate.train --curves experiments\data\curves.jsonl
}

Write-Host "Starting dashboard server in the background (survives this window closing)..."
$proc = Start-Process -FilePath $VenvPython `
    -ArgumentList "-m", "uvicorn", "dashboard.backend.app:app", "--host", "0.0.0.0", "--port", $Port `
    -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err" `
    -WindowStyle Hidden -PassThru
$proc.Id | Out-File -FilePath $PidFile -Encoding ascii
Start-Sleep -Seconds 2

if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) {
    Write-Host ""
    Write-Host "Dashboard is running (pid $($proc.Id)), logging to $LogFile"
    Write-Host "Open http://localhost:$Port in a browser any time - training keeps going without it open."
    Write-Host "To stop it: Stop-Process -Id (Get-Content dashboard.pid)"
} else {
    Write-Host "Server failed to start - check $LogFile"
    exit 1
}
