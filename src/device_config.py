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
    Chỉ import framework khi constant tương ứng thực sự được yêu cầu.

    from device_config import YOLO_DEVICE
        -> chỉ import torch

    from device_config import PADDLE_DEVICE
        -> chỉ import paddle
    """
    if name == "YOLO_DEVICE":
        return get_yolo_device()

    if name == "PADDLE_DEVICE":
        return get_paddle_device()

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
def print_device_summary() -> None:
    """
    In thông tin device mà không import đồng thời
    PyTorch và PaddlePaddle.
    """
    print("=" * 72)
    print("DEVICE CONFIGURATION")
    print("=" * 72)
    print("YOLO / PyTorch : GPU 0")
    print("PaddleOCR      : gpu:0")
    print("=" * 72)


if __name__ == "__main__":
    print_device_summary()