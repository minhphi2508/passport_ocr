from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "viz_stage"
    / "color"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "viz_stage"
    / "enhanced"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff",
}

MIN_WIDTH = 1400


def resize_if_needed(
    image: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]

    if width >= MIN_WIDTH:
        return image

    scale = MIN_WIDTH / width

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC,
    )


def enhance_viz(
    image: np.ndarray,
) -> np.ndarray:
    image = resize_if_needed(image)

    # Chuyển sang LAB để chỉnh độ sáng mà không phá màu.
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8),
    )

    l_enhanced = clahe.apply(
        l_channel
    )

    enhanced_lab = cv2.merge(
        (
            l_enhanced,
            a_channel,
            b_channel,
        )
    )

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )

    # Sharpen rất nhẹ.
    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        sigmaX=1.0,
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.15,
        blurred,
        -0.15,
        0,
    )

    return sharpened


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Không thấy input folder:\n{INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        path
        for path in INPUT_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            "Không tìm thấy ảnh VIZ."
        )

    success = 0
    errors = 0

    print(
        f"Tổng ảnh VIZ: "
        f"{len(image_paths)}"
    )

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            errors += 1

            print(
                f"[{index:>3}/{len(image_paths)}] "
                f"{image_path.name} "
                f"-> read_error"
            )

            continue

        enhanced = enhance_viz(
            image
        )

        output_path = (
            OUTPUT_DIR
            / image_path.name
        )

        saved = cv2.imwrite(
            str(output_path),
            enhanced,
        )

        if saved:
            success += 1
            status = "success"

        else:
            errors += 1
            status = "save_error"

        print(
            f"[{index:>3}/{len(image_paths)}] "
            f"{image_path.name} "
            f"-> {status}"
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "KẾT QUẢ PREPROCESS VIZ"
    )

    print(
        "=" * 72
    )

    print(
        f"Tổng ảnh          : "
        f"{len(image_paths)}"
    )

    print(
        f"Thành công        : "
        f"{success}"
    )

    print(
        f"Lỗi               : "
        f"{errors}"
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()