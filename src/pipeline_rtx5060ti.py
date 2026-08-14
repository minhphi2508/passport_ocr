from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pipeline as base_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TORCH_PYTHON = (
    PROJECT_ROOT
    / ".venv-torch"
    / "Scripts"
    / "python.exe"
)

PADDLE_PYTHON = (
    PROJECT_ROOT
    / ".venv-paddle"
    / "Scripts"
    / "python.exe"
)

# These stages load PaddleOCR/PaddleX and must be isolated from
# the PyTorch CUDA DLL set on Windows.
PADDLE_SCRIPTS = {
    "ocr_mrz_batch.py",
    "ocr_viz_batch.py",
}


def _require_python(
    path: Path,
    label: str,
) -> None:
    if path.exists():
        return

    raise FileNotFoundError(
        f"Không thấy {label} environment:\n"
        f"{path}\n\n"
        "Hãy chạy trước:\n"
        "  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass\n"
        "  .\\scripts\\setup_rtx5060ti.ps1"
    )


def _probe(
    python_executable: Path,
    code: str,
) -> str:
    result = subprocess.run(
        [
            str(python_executable),
            "-c",
            code,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()

        if stderr:
            return (
                "unavailable "
                f"({stderr.splitlines()[-1]})"
            )

        return f"unavailable (exit {result.returncode})"

    stdout = result.stdout.strip()

    if not stdout:
        return "unknown"

    return stdout.splitlines()[-1]


def print_device_summary() -> None:
    _require_python(
        TORCH_PYTHON,
        "PyTorch",
    )
    _require_python(
        PADDLE_PYTHON,
        "PaddleOCR",
    )

    torch_summary = _probe(
        TORCH_PYTHON,
        r"""
import torch

if not torch.cuda.is_available():
    print("cpu")
else:
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(
        f"GPU 0 ({name}, "
        f"sm_{capability[0]}{capability[1]}, "
        f"torch={torch.__version__}, "
        f"cuda={torch.version.cuda})"
    )
""",
    )

    paddle_summary = _probe(
        PADDLE_PYTHON,
        r"""
import paddle

if (
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0
):
    paddle.set_device("gpu:0")
    print(
        f"gpu:0 "
        f"(paddle={paddle.__version__}, "
        f"gpu_count={paddle.device.cuda.device_count()})"
    )
else:
    print("cpu")
""",
    )

    print("=" * 72)
    print("DEVICE CONFIGURATION - RTX 5060 Ti")
    print("=" * 72)
    print(f"YOLO / PyTorch : {torch_summary}")
    print(f"PaddleOCR      : {paddle_summary}")
    print()
    print(f"Torch Python   : {TORCH_PYTHON}")
    print(f"Paddle Python  : {PADDLE_PYTHON}")
    print("=" * 72)


def run_stage(
    stage_number: int,
    total_stages: int,
    name: str,
    script: str,
) -> None:
    _require_python(
        TORCH_PYTHON,
        "PyTorch",
    )
    _require_python(
        PADDLE_PYTHON,
        "PaddleOCR",
    )

    script_path = (
        base_pipeline.SRC_DIR
        / script
    )

    if script in PADDLE_SCRIPTS:
        python_executable = PADDLE_PYTHON
        backend = "Paddle GPU"
    else:
        python_executable = TORCH_PYTHON
        backend = "Torch/common"

    print()
    print("=" * 76)
    print(
        f"[{stage_number}/"
        f"{total_stages}] "
        f"{name}"
    )
    print("=" * 76)
    print(f"Script : {script}")
    print(f"Backend: {backend}")
    print(f"Python : {python_executable}")

    start_time = time.time()

    result = subprocess.run(
        [
            str(python_executable),
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print()
        print("=" * 76)
        print("PIPELINE DỪNG")
        print("=" * 76)
        print(f"Stage lỗi : {name}")
        print(f"Script     : {script}")
        print(f"Backend    : {backend}")
        print(f"Python     : {python_executable}")
        print(f"Return code: {result.returncode}")

        raise RuntimeError(
            f"Stage failed: {name}"
        )

    print()
    print(f"✓ Hoàn thành: {name}")
    print(f"Thời gian   : {elapsed:.1f}s")


# Reuse the existing CLI, stage list, checkpointing and output logic.
# Only the process interpreter routing and device summary are replaced.
base_pipeline.run_stage = run_stage
base_pipeline.print_device_summary = print_device_summary


if __name__ == "__main__":
    base_pipeline.main()
