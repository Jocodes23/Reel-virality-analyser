<#
  run.ps1 — Windows helper for reels-trend-intel.

  Usage:
    ./run.ps1 setup        # create venv + install (SQLite path, no GPU extras)
    ./run.ps1 setup-gpu    # also install the real ML stack (torch/CLIP/audio/OCR)
    ./run.ps1 golden       # run the offline golden path end-to-end (+ throughput report)
    ./run.ps1 serve        # start the read-only API + dashboard at http://127.0.0.1:8000
    ./run.ps1 loop         # start the resilient production loop
    ./run.ps1 test         # run the test suite
#>
param([Parameter(Position = 0)][string]$cmd = "help")

$ErrorActionPreference = "Stop"
$venv = ".venv"
$py = Join-Path $venv "Scripts\python.exe"

function Ensure-Venv {
    if (-not (Test-Path $py)) {
        Write-Host "Creating virtual environment..." -ForegroundColor Cyan
        python -m venv $venv
    }
}

switch ($cmd) {
    "setup" {
        Ensure-Venv
        & $py -m pip install --upgrade pip
        & $py -m pip install -e ".[dev]"
        Write-Host "Setup complete (SQLite backend, deterministic embeddings)." -ForegroundColor Green
    }
    "setup-gpu" {
        Ensure-Venv
        & $py -m pip install --upgrade pip
        & $py -m pip install -e ".[dev,gpu,audio,ocr,clustering]"
        Write-Host "GPU stack installed. Set RTI_FEATURES__EMBEDDING_BACKEND=real and" -ForegroundColor Green
        Write-Host "RTI_FEATURES__DEVICE=cuda to use the real models (CPU fallback is automatic)." -ForegroundColor Green
    }
    "golden" {
        Ensure-Venv
        & $py -m reels_trend_intel.cli run-once --step 30
    }
    "serve" { Ensure-Venv; & $py -m reels_trend_intel.cli serve }
    "loop"  { Ensure-Venv; & $py -m reels_trend_intel.cli loop }
    "test"  { Ensure-Venv; & $py -m pytest -q }
    default {
        Write-Host "Commands: setup | setup-gpu | golden | serve | loop | test"
    }
}
