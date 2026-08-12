#requires -Version 5.1
$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "Passport OCR - CPU setup"
Write-Host "Python 3.12"
Write-Host "============================================================"

if (-not (Test-Path ".git")) {
    Write-Warning "Run this script from the repository root."
}

if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-cpu.txt

Write-Host ""
Write-Host "Dependency check"
python -m pip check

Write-Host ""
Write-Host "CPU backend verification"
python -c "import torch; print('Torch:', torch.__version__); print('Torch CUDA available:', torch.cuda.is_available())"
python -c "import paddle; print('Paddle:', paddle.__version__); print('Paddle CUDA compiled:', paddle.device.is_compiled_with_cuda())"

Write-Host ""
Write-Host "Repository device configuration"
python src/device_config.py

Write-Host ""
Write-Host "CPU setup completed."
