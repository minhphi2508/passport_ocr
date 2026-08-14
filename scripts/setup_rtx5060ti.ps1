#requires -Version 5.1
$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "Passport OCR - RTX 5060 Ti setup"
Write-Host "Dual isolated GPU environments"
Write-Host "PyTorch CUDA 12.8 / Paddle CUDA 12.9"
Write-Host "============================================================"

if (-not (Test-Path ".git")) {
    throw "Run this script from the repository root."
}

$TorchEnv = ".venv-torch"
$PaddleEnv = ".venv-paddle"

$TorchPython = Join-Path $TorchEnv "Scripts\python.exe"
$PaddlePython = Join-Path $PaddleEnv "Scripts\python.exe"

Write-Host ""
Write-Host "[1/8] Checking Python 3.12..."
py -3.12 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 is required."
}

Write-Host ""
Write-Host "[2/8] Creating isolated virtual environments..."

if (-not (Test-Path $TorchPython)) {
    Write-Host "Creating $TorchEnv..."
    py -3.12 -m venv $TorchEnv
} else {
    Write-Host "$TorchEnv already exists."
}

if (-not (Test-Path $PaddlePython)) {
    Write-Host "Creating $PaddleEnv..."
    py -3.12 -m venv $PaddleEnv
} else {
    Write-Host "$PaddleEnv already exists."
}

Write-Host ""
Write-Host "[3/8] Updating pip/setuptools/wheel..."
& $TorchPython -m pip install --upgrade pip setuptools wheel
& $PaddlePython -m pip install --upgrade pip setuptools wheel

Write-Host ""
Write-Host "[4/8] Installing PyTorch / YOLO environment..."

& $TorchPython -m pip install `
    numpy==2.3.5 `
    opencv-python==5.0.0.93 `
    ultralytics==8.4.114

$ErrorActionPreference = "Continue"
& $TorchPython -m pip uninstall -y torch torchvision torchaudio
$ErrorActionPreference = "Stop"

& $TorchPython -m pip install `
    torch==2.8.0 `
    torchvision==0.23.0 `
    --index-url https://download.pytorch.org/whl/cu128

Write-Host ""
Write-Host "[5/8] Installing PaddleOCR environment..."

& $PaddlePython -m pip install `
    numpy==2.3.5 `
    opencv-python==5.0.0.93 `
    paddleocr==3.7.0 `
    paddlex==3.7.2

$ErrorActionPreference = "Continue"
& $PaddlePython -m pip uninstall -y paddlepaddle paddlepaddle-gpu
$ErrorActionPreference = "Stop"

& $PaddlePython -m pip install `
    paddlepaddle-gpu==3.3.0 `
    -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

Write-Host ""
Write-Host "[6/8] Checking dependencies..."
& $TorchPython -m pip check
& $PaddlePython -m pip check

Write-Host ""
Write-Host "[7/8] PyTorch GPU smoke test..."
& $TorchPython -c "import torch; assert torch.cuda.is_available(); x=torch.randn(1024,1024,device='cuda'); y=x@x; print('Torch:',torch.__version__); print('CUDA runtime:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0)); print('Capability:',torch.cuda.get_device_capability(0)); print('Torch GPU test:',float(y.mean()))"

Write-Host ""
Write-Host "Paddle GPU smoke test..."
& $PaddlePython -c "import paddle; assert paddle.device.is_compiled_with_cuda(); assert paddle.device.cuda.device_count()>0; paddle.set_device('gpu:0'); x=paddle.randn([1024,1024]); y=paddle.matmul(x,x); print('Paddle:',paddle.__version__); print('GPU count:',paddle.device.cuda.device_count()); print('Device:',paddle.device.get_device()); print('Paddle GPU test:',float(paddle.mean(y)))"

Write-Host ""
Write-Host "PaddleOCR import test..."
& $PaddlePython -c "from paddleocr import PaddleOCR; import paddle; paddle.set_device('gpu:0'); print('PaddleOCR import: OK'); print('Device:',paddle.device.get_device())"

Write-Host ""
Write-Host "[8/8] Repository device health check..."
py -3.12 src\device_config.py

Write-Host ""
Write-Host "============================================================"
Write-Host "RTX 5060 Ti setup completed."
Write-Host "============================================================"
Write-Host ""
Write-Host "Torch environment : $TorchPython"
Write-Host "Paddle environment: $PaddlePython"
Write-Host ""
Write-Host "Run:"
Write-Host "  .\scripts\run_rtx5060ti.ps1 --fresh"
