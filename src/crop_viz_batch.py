from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CẤU HÌNH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "passport_detector_ver3_best.pt"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "passport_pages_safe"
    / "transformed"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "viz_stage"

COLOR_DIR = OUTPUT_ROOT / "color"
GRAYSCALE_DIR = OUTPUT_ROOT / "grayscale"
DEBUG_DIR = OUTPUT_ROOT / "debug"
CSV_PATH = OUTPUT_ROOT / "viz_crop_results.csv"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff",
}

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

# Padding quanh MRZ trước khi cắt VIZ.
MRZ_TOP_MARGIN_RATIO = 0.04

# Nếu không detect được MRZ, fallback:
# lấy 78% phía trên của passport page làm VIZ.
FALLBACK_VIZ_HEIGHT_RATIO = 0.78


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================

def safe_output_name(
    image_path: Path,
    input_root: Path,
) -> str:
    relative_path = image_path.relative_to(input_root)
    parts = list(relative_path.parts)

    stem = Path(parts[-1]).stem
    safe_stem = "__".join(parts[:-1] + [stem])

    return f"{safe_stem}.jpg"


def create_grayscale_variant(
    viz_crop: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(
        viz_crop,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=1.6,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


# ============================================================
# DETECT MRZ
# ============================================================

def detect_best_mrz(
    model: YOLO,
    image_path: Path,
) -> dict[str, object] | None:
    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        device="cpu",
        save=False,
        verbose=False,
    )[0]

    candidates: list[dict[str, object]] = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = result.names[class_id]

            if class_name != "mrz":
                continue

            confidence = float(box.conf.item())

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .tolist()
            )

            candidates.append(
                {
                    "confidence": confidence,
                    "bbox": (
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ),
                }
            )

    del result

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item["confidence"],
    )


# ============================================================
# XỬ LÝ MỘT ẢNH
# ============================================================

def process_one_image(
    model: YOLO,
    image_path: Path,
) -> dict[str, object]:
    output_name = safe_output_name(
        image_path,
        INPUT_DIR,
    )

    color_path = COLOR_DIR / output_name
    grayscale_path = GRAYSCALE_DIR / output_name
    debug_path = DEBUG_DIR / output_name

    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "filename": image_path.name,
            "status": "image_read_failed",
            "mrz_detected": False,
            "viz_mode": None,
            "error": "OpenCV không đọc được ảnh.",
        }

    height, width = image.shape[:2]

    mrz = detect_best_mrz(
        model=model,
        image_path=image_path,
    )

    debug_image = image.copy()

    if mrz is not None:
        x1, y1, x2, y2 = mrz["bbox"]

        # Cắt VIZ ngay phía trên MRZ,
        # chừa một khoảng an toàn.
        margin = int(
            round(height * MRZ_TOP_MARGIN_RATIO)
        )

        viz_bottom = max(
            1,
            y1 - margin,
        )

        viz_crop = image[
            0:viz_bottom,
            0:width,
        ]

        viz_mode = "mrz_based"
        mrz_detected = True

        cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        cv2.line(
            debug_image,
            (0, viz_bottom),
            (width - 1, viz_bottom),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            debug_image,
            f"MRZ {mrz['confidence']:.3f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    else:
        viz_bottom = int(
            round(
                height
                * FALLBACK_VIZ_HEIGHT_RATIO
            )
        )

        viz_crop = image[
            0:viz_bottom,
            0:width,
        ]

        viz_mode = "ratio_fallback"
        mrz_detected = False

        cv2.line(
            debug_image,
            (0, viz_bottom),
            (width - 1, viz_bottom),
            (0, 255, 255),
            3,
        )

    if viz_crop.size == 0:
        return {
            "filename": image_path.name,
            "status": "empty_viz_crop",
            "mrz_detected": mrz_detected,
            "viz_mode": viz_mode,
            "error": "VIZ crop rỗng.",
        }

    grayscale = create_grayscale_variant(
        viz_crop
    )

    cv2.imwrite(
        str(color_path),
        viz_crop,
    )

    cv2.imwrite(
        str(grayscale_path),
        grayscale,
    )

    cv2.imwrite(
        str(debug_path),
        debug_image,
    )

    return {
        "filename": image_path.name,
        "status": "success",
        "mrz_detected": mrz_detected,
        "viz_mode": viz_mode,
        "mrz_confidence": (
            round(
                float(mrz["confidence"]),
                6,
            )
            if mrz is not None
            else None
        ),
        "original_width": width,
        "original_height": height,
        "viz_width": viz_crop.shape[1],
        "viz_height": viz_crop.shape[0],
        "viz_bottom_y": viz_bottom,
        "color_path": str(color_path),
        "grayscale_path": str(
            grayscale_path
        ),
        "debug_path": str(debug_path),
        "error": None,
    }


# ============================================================
# CSV
# ============================================================

def write_csv(
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không thấy model:\n{MODEL_PATH}"
        )

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Không thấy input folder:\n{INPUT_DIR}"
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
            f"Không có ảnh trong:\n{INPUT_DIR}"
        )

    for directory in (
        OUTPUT_ROOT,
        COLOR_DIR,
        GRAYSCALE_DIR,
        DEBUG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        f"Tổng passport pages: "
        f"{len(image_paths)}"
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    rows: list[dict[str, object]] = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        try:
            row = process_one_image(
                model=model,
                image_path=image_path,
            )

        except Exception as error:
            row = {
                "filename": image_path.name,
                "status": "unexpected_error",
                "mrz_detected": False,
                "viz_mode": None,
                "error": repr(error),
            }

        rows.append(row)

        write_csv(rows)

        print(
            f"[{index:>3}/{len(image_paths)}] "
            f"{image_path.name} "
            f"-> {row['status']} "
            f"| {row.get('viz_mode')}"
        )

    success_count = sum(
        1
        for row in rows
        if row["status"] == "success"
    )

    mrz_based_count = sum(
        1
        for row in rows
        if row.get("viz_mode") == "mrz_based"
    )

    fallback_count = sum(
        1
        for row in rows
        if row.get("viz_mode")
        == "ratio_fallback"
    )

    print("\n" + "=" * 72)
    print("KẾT QUẢ CROP VIZ")
    print("=" * 72)

    print(
        f"Tổng ảnh                 : "
        f"{len(rows)}"
    )

    print(
        f"Crop thành công           : "
        f"{success_count}"
    )

    print(
        f"MRZ-based crop            : "
        f"{mrz_based_count}"
    )

    print(
        f"Ratio fallback             : "
        f"{fallback_count}"
    )

    print("\nColor:")
    print(COLOR_DIR)

    print("\nGrayscale:")
    print(GRAYSCALE_DIR)

    print("\nDebug:")
    print(DEBUG_DIR)

    print("\nCSV:")
    print(CSV_PATH)


if __name__ == "__main__":
    main()