from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from device_config import YOLO_DEVICE

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

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "mrz_stage"

ORIGINAL_DIR = OUTPUT_ROOT / "original_crops"
GRAYSCALE_DIR = OUTPUT_ROOT / "grayscale"
THRESHOLD_DIR = OUTPUT_ROOT / "threshold"
DEBUG_DIR = OUTPUT_ROOT / "debug"
CSV_PATH = OUTPUT_ROOT / "mrz_detection_results.csv"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff",
}

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

# MRZ nằm sát mép nên padding phải vừa phải.
PADDING_X_RATIO = 0.015
PADDING_Y_RATIO = 0.18

# Nâng chiều cao ảnh MRZ để OCR dễ hơn.
TARGET_HEIGHT = 180


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


def add_padding(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    box_width = x2 - x1
    box_height = y2 - y1

    padding_x = int(round(box_width * PADDING_X_RATIO))
    padding_y = int(round(box_height * PADDING_Y_RATIO))

    return (
        max(0, x1 - padding_x),
        max(0, y1 - padding_y),
        min(image_width, x2 + padding_x),
        min(image_height, y2 + padding_y),
    )


def resize_to_target_height(
    image: np.ndarray,
    target_height: int,
) -> np.ndarray:
    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError("Ảnh crop có kích thước không hợp lệ.")

    scale = target_height / height
    target_width = max(1, int(round(width * scale)))

    interpolation = (
        cv2.INTER_CUBIC
        if scale > 1
        else cv2.INTER_AREA
    )

    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=interpolation,
    )


def create_grayscale_variant(
    mrz_crop: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(
        mrz_crop,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(8, 4),
    )

    enhanced = clahe.apply(gray)

    return resize_to_target_height(
        enhanced,
        TARGET_HEIGHT,
    )


def create_threshold_variant(
    grayscale_image: np.ndarray,
) -> np.ndarray:
    # Otsu tự tìm threshold theo từng ảnh.
    _, thresholded = cv2.threshold(
        grayscale_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Đảm bảo nền chủ yếu là trắng.
    white_ratio = float(
        np.mean(thresholded == 255)
    )

    if white_ratio < 0.50:
        thresholded = cv2.bitwise_not(
            thresholded
        )

    return thresholded


# ============================================================
# DETECT MRZ
# ============================================================

def detect_mrz_candidates(
    model: YOLO,
    image_path: Path,
) -> list[dict[str, object]]:
    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=YOLO_DEVICE,
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

    return candidates


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

    original_path = ORIGINAL_DIR / output_name
    grayscale_path = GRAYSCALE_DIR / output_name
    threshold_path = THRESHOLD_DIR / output_name
    debug_path = DEBUG_DIR / output_name

    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "filename": image_path.name,
            "relative_path": str(
                image_path.relative_to(INPUT_DIR)
            ),
            "status": "image_read_failed",
            "mrz_count": 0,
            "selected_confidence": None,
            "error": "OpenCV không đọc được ảnh.",
        }

    image_height, image_width = image.shape[:2]

    candidates = detect_mrz_candidates(
        model=model,
        image_path=image_path,
    )

    if not candidates:
        return {
            "filename": image_path.name,
            "relative_path": str(
                image_path.relative_to(INPUT_DIR)
            ),
            "status": "mrz_not_detected",
            "mrz_count": 0,
            "selected_confidence": None,
            "error": None,
        }

    best_candidate = max(
        candidates,
        key=lambda item: item["confidence"],
    )

    raw_bbox = best_candidate["bbox"]

    x1, y1, x2, y2 = add_padding(
        bbox=raw_bbox,
        image_width=image_width,
        image_height=image_height,
    )

    mrz_crop = image[y1:y2, x1:x2]

    if mrz_crop.size == 0:
        return {
            "filename": image_path.name,
            "relative_path": str(
                image_path.relative_to(INPUT_DIR)
            ),
            "status": "empty_mrz_crop",
            "mrz_count": len(candidates),
            "selected_confidence": best_candidate["confidence"],
            "error": "MRZ crop rỗng.",
        }

    # Bản màu gốc được resize nhưng không enhance.
    original_resized = resize_to_target_height(
        mrz_crop,
        TARGET_HEIGHT,
    )

    grayscale = create_grayscale_variant(
        mrz_crop
    )

    thresholded = create_threshold_variant(
        grayscale
    )

    cv2.imwrite(
        str(original_path),
        original_resized,
    )

    cv2.imwrite(
        str(grayscale_path),
        grayscale,
    )

    cv2.imwrite(
        str(threshold_path),
        thresholded,
    )

    debug_image = image.copy()

    # Vẽ toàn bộ candidate:
    # candidate được chọn màu xanh, các candidate thừa màu đỏ.
    for candidate in candidates:
        bx1, by1, bx2, by2 = candidate["bbox"]
        selected = candidate is best_candidate

        color = (
            (0, 255, 0)
            if selected
            else (0, 0, 255)
        )

        cv2.rectangle(
            debug_image,
            (bx1, by1),
            (bx2, by2),
            color,
            3,
        )

        cv2.putText(
            debug_image,
            f"MRZ {candidate['confidence']:.3f}",
            (bx1, max(25, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(debug_path),
        debug_image,
    )

    status = (
        "success_single_mrz"
        if len(candidates) == 1
        else "success_multiple_mrz_selected_best"
    )

    return {
        "filename": image_path.name,
        "relative_path": str(
            image_path.relative_to(INPUT_DIR)
        ),
        "status": status,
        "mrz_count": len(candidates),
        "selected_confidence": round(
            float(best_candidate["confidence"]),
            6,
        ),
        "selected_bbox_x1": raw_bbox[0],
        "selected_bbox_y1": raw_bbox[1],
        "selected_bbox_x2": raw_bbox[2],
        "selected_bbox_y2": raw_bbox[3],
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        "crop_width": mrz_crop.shape[1],
        "crop_height": mrz_crop.shape[0],
        "output_width": original_resized.shape[1],
        "output_height": original_resized.shape[0],
        "original_crop_path": str(original_path),
        "grayscale_path": str(grayscale_path),
        "threshold_path": str(threshold_path),
        "debug_path": str(debug_path),
        "error": None,
    }


# ============================================================
# CSV
# ============================================================

def write_csv(
    rows: list[dict[str, object]],
) -> None:
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
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"Không có ảnh trong:\n{INPUT_DIR}"
        )

    for directory in (
        OUTPUT_ROOT,
        ORIGINAL_DIR,
        GRAYSCALE_DIR,
        THRESHOLD_DIR,
        DEBUG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(f"Tổng ảnh passport page: {len(image_paths)}")
    print(f"Input: {INPUT_DIR}")
    print()

    model = YOLO(str(MODEL_PATH))
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
                "relative_path": str(
                    image_path.relative_to(INPUT_DIR)
                ),
                "status": "unexpected_error",
                "mrz_count": None,
                "selected_confidence": None,
                "error": repr(error),
            }

        rows.append(row)
        write_csv(rows)

        print(
            f"[{index:>4}/{len(image_paths)}] "
            f"{row['relative_path']} "
            f"-> {row['status']}"
        )

    status_counts: dict[str, int] = {}

    for row in rows:
        status = str(row["status"])
        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    print("\n" + "=" * 72)
    print("KẾT QUẢ CROP MRZ")
    print("=" * 72)

    for status, count in sorted(
        status_counts.items()
    ):
        print(f"{status:<40}: {count}")

    selected_confidences = [
        float(row["selected_confidence"])
        for row in rows
        if row.get("selected_confidence") is not None
    ]

    if selected_confidences:
        print(
            f"\nConfidence trung bình: "
            f"{np.mean(selected_confidences):.4f}"
        )

        print(
            f"Confidence thấp nhất : "
            f"{np.min(selected_confidences):.4f}"
        )

    print("\nOriginal MRZ:")
    print(ORIGINAL_DIR)

    print("\nGrayscale MRZ:")
    print(GRAYSCALE_DIR)

    print("\nThreshold MRZ:")
    print(THRESHOLD_DIR)

    print("\nDebug:")
    print(DEBUG_DIR)

    print("\nCSV:")
    print(CSV_PATH)


if __name__ == "__main__":
    main()