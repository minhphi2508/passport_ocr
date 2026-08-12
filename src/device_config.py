from __future__ import annotations


# ============================================================
# YOLO / PYTORCH DEVICE
# ============================================================

def get_yolo_device() -> int | str:
    """
    Ultralytics / PyTorch device.

    GPU available:
        0

    Otherwise:
        "cpu"
    """
    import torch

    if torch.cuda.is_available():
        return 0

    return "cpu"


# ============================================================
# PADDLEOCR DEVICE
# ============================================================

def get_paddle_device() -> str:
    """
    PaddleOCR device.

    GPU-enabled PaddlePaddle + CUDA available:
        "gpu:0"

    Otherwise:
        "cpu"
    """
    import paddle

    try:
        if (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        ):
            return "gpu:0"
    except Exception:
        pass

    return "cpu"


# ============================================================
# LAZY DEVICE CONSTANTS
# ============================================================

def __getattr__(name: str):
    """
    Import only the framework required by the requested constant.

    from device_config import YOLO_DEVICE
        -> imports torch only

    from device_config import PADDLE_DEVICE
        -> imports paddle only
    """
    if name == "YOLO_DEVICE":
        return get_yolo_device()

    if name == "PADDLE_DEVICE":
        return get_paddle_device()

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def _get_torch_summary() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"

        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        return f"GPU 0 ({name}, sm_{capability[0]}{capability[1]})"
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def _get_paddle_summary() -> str:
    try:
        import paddle

        if (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        ):
            return "gpu:0"

        return "cpu"
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def print_device_summary() -> None:
    """
    Print actual backend/device detection.

    Torch and Paddle are imported independently so one broken framework
    does not prevent reporting the state of the other framework.
    """
    print("=" * 72)
    print("DEVICE CONFIGURATION")
    print("=" * 72)
    print(f"YOLO / PyTorch : {_get_torch_summary()}")
    print(f"PaddleOCR      : {_get_paddle_summary()}")
    print("=" * 72)


if __name__ == "__main__":
    print_device_summary()
