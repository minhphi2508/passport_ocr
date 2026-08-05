from __future__ import annotations

import torch
import paddle


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
# DEVICE SUMMARY
# ============================================================

YOLO_DEVICE = get_yolo_device()
PADDLE_DEVICE = get_paddle_device()


def print_device_summary() -> None:
    print("=" * 72)
    print("DEVICE CONFIGURATION")
    print("=" * 72)

    if YOLO_DEVICE == 0:
        print(
            f"YOLO / PyTorch : GPU 0 "
            f"({torch.cuda.get_device_name(0)})"
        )
    else:
        print("YOLO / PyTorch : CPU")

    print(
        f"PaddleOCR      : "
        f"{PADDLE_DEVICE}"
    )

    print("=" * 72)

if __name__ == "__main__":
    print_device_summary()