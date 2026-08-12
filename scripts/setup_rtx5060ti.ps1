#requires -Version 5.1
$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "Passport OCR - RTX 5060 Ti setup"
Write-Host "Python 3.12 / PyTorch CUDA 12.8 / Paddle CUDA 12.9"
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

Write-Host ""
Write-Host "[1/4] Installing shared dependencies..."
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "[2/4] Installing PyTorch 2.8.0 / CUDA 12.8..."
python -m pip uninstall -y torch torchvision torchaudio 2>$null
python -m pip install `
    torch==2.8.0 `
    torchvision==0.23.0 `
    --index-url https://download.pytorch.org/whl/cu128

Write-Host ""
Write-Host "[3/4] Installing PaddlePaddle GPU 3.3.0 / CUDA 12.9..."
python -m pip uninstall -y paddlepaddle paddlepaddle-gpu 2>$null
python -m pip install `
    paddlepaddle-gpu==3.3.0 `
    -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

Write-Host ""
Write-Host "[4/4] Verifying environment..."

python -m pip check

Write-Host ""
Write-Host "PyTorch GPU smoke test"
python -c "import torch; print('Torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else ''); x=torch.randn(1024,1024,device='cuda'); y=x@x; print('Torch GPU test:', float(y.mean()))"

Write-Host ""
Write-Host "Paddle GPU smoke test"
python -c "import paddle; print('Paddle:', paddle.__version__); print('Compiled CUDA:', paddle.device.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count()); paddle.set_device('gpu:0'); x=paddle.randn([1024,1024]); y=paddle.matmul(x,x); print('Device:', paddle.device.get_device()); print('Paddle GPU test:', float(paddle.mean(y)))"

Write-Host ""
Write-Host "Repository device configuration"
python src/device_config.py

Write-Host ""
Write-Host "RTX 5060 Ti setup completed."
Write-Host "Place images in .\input_images\ and run:"
Write-Host "  python src/pipeline.py --fresh"
