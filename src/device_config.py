from __future__ import annotations

import subprocess
from pathlib import Path


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


def get_yolo_device() -> int | str:
    """Return the Ultralytics device inside the Torch environment."""
    import torch

    if torch.cuda.is_available():
        return 0

    return "cpu"


def get_paddle_device() -> str:
    """Return the PaddleOCR device inside the Paddle environment."""
    import paddle

    if (
        paddle.device.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
    ):
        return "gpu:0"

    return "cpu"


def __getattr__(name: str):
    """
    Lazy constants used by the existing stage scripts.

    YOLO_DEVICE is evaluated only inside Torch-routed stages.
    PADDLE_DEVICE is evaluated only inside Paddle-routed stages.
    """
    if name == "YOLO_DEVICE":
        return get_yolo_device()

    if name == "PADDLE_DEVICE":
        return get_paddle_device()

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def _probe(
    python_executable: Path,
    code: str,
) -> str:
    """Probe one framework in its own process to avoid CUDA DLL collisions."""
    if not python_executable.exists():
        return f"environment not found ({python_executable})"

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

    # Paddle may print informational lines before our final result.
    return stdout.splitlines()[-1]


def _get_torch_summary() -> str:
    return _probe(
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


def _get_paddle_summary() -> str:
    return _probe(
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


def print_device_summary() -> None:
    print("=" * 72)
    print("DEVICE CONFIGURATION - DUAL GPU ENV")
    print("=" * 72)
    print(f"YOLO / PyTorch : {_get_torch_summary()}")
    print(f"PaddleOCR      : {_get_paddle_summary()}")
    print()
    print(f"Torch Python   : {TORCH_PYTHON}")
    print(f"Paddle Python  : {PADDLE_PYTHON}")
    print("=" * 72)


if __name__ == "__main__":
    print_device_summary()
