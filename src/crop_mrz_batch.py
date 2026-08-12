from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGE_ROOT = PROJECT_ROOT / "outputs" / "passport_pages_safe"
PROCESSING_CSV = PAGE_ROOT / "processing_results.csv"
INPUT_DIR = PAGE_ROOT / "transformed"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "mrz_stage"
ORIGINAL_DIR = OUTPUT_ROOT / "original_crops"
GRAYSCALE_DIR = OUTPUT_ROOT / "grayscale"
THRESHOLD_DIR = OUTPUT_ROOT / "threshold"
DEBUG_DIR = OUTPUT_ROOT / "debug"
CSV_PATH = OUTPUT_ROOT / "mrz_detection_results.csv"

PADDING_X_RATIO = 0.015
PADDING_Y_RATIO = 0.18
TARGET_HEIGHT = 180


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def parse_int(value: Any) -> int | None:
    if value in (None, "", "None", "null"):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def load_processing_rows() -> list[dict[str, str]]:
    if not PROCESSING_CSV.exists():
        raise FileNotFoundError(
            f"Không thấy passport processing CSV:\n{PROCESSING_CSV}"
        )

    with PROCESSING_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


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


def resize_to_target_height(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Ảnh crop có kích thước không hợp lệ.")

    scale = target_height / height
    target_width = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=interpolation,
    )


def create_grayscale_variant(mrz_crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(mrz_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 4))
    enhanced = clahe.apply(gray)
    return resize_to_target_height(enhanced, TARGET_HEIGHT)


def create_threshold_variant(grayscale_image: np.ndarray) -> np.ndarray:
    _, thresholded = cv2.threshold(
        grayscale_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    if float(np.mean(thresholded == 255)) < 0.50:
        thresholded = cv2.bitwise_not(thresholded)

    return thresholded


def metadata_bbox(row: dict[str, str]) -> tuple[int, int, int, int] | None:
    if not parse_bool(row.get("transformed_mrz_detected")):
        return None

    values = [
        parse_int(row.get("transformed_mrz_x1")),
        parse_int(row.get("transformed_mrz_y1")),
        parse_int(row.get("transformed_mrz_x2")),
        parse_int(row.get("transformed_mrz_y2")),
    ]

    if any(value is None for value in values):
        return None

    x1, y1, x2, y2 = values
    assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
    return x1, y1, x2, y2


def process_one_row(row: dict[str, str]) -> dict[str, Any]:
    sample_id = row.get("sample_id") or ""
    source_filename = row.get("source_filename") or ""
    relative_path = row.get("relative_path") or ""
    generated_filename = row.get("generated_filename") or f"{sample_id}.jpg"

    base = {
        "sample_id": sample_id,
        "filename": generated_filename,
        "source_filename": source_filename,
        "relative_path": relative_path,
    }

    transformed_path_text = row.get("transformed_path")
    bbox = metadata_bbox(row)

    if not transformed_path_text:
        return {
            **base,
            "status": "passport_page_unavailable",
            "mrz_count": 0,
            "selected_confidence": None,
            "error": None,
        }

    image_path = Path(transformed_path_text)
    if not image_path.exists():
        image_path = INPUT_DIR / generated_filename

    image = cv2.imread(str(image_path))
    if image is None:
        return {
            **base,
            "status": "image_read_failed",
            "mrz_count": 0,
            "selected_confidence": None,
            "error": f"Không đọc được transformed page: {image_path}",
        }

    if bbox is None:
        return {
            **base,
            "status": "mrz_not_detected",
            "mrz_count": 0,
            "selected_confidence": None,
            "error": None,
        }

    image_height, image_width = image.shape[:2]
    raw_bbox = bbox
    x1, y1, x2, y2 = add_padding(
        bbox=raw_bbox,
        image_width=image_width,
        image_height=image_height,
    )

    mrz_crop = image[y1:y2, x1:x2]
    if mrz_crop.size == 0:
        return {
            **base,
            "status": "empty_mrz_crop",
            "mrz_count": 1,
            "selected_confidence": parse_float(
                row.get("transformed_mrz_confidence")
            ),
            "error": "MRZ crop rỗng.",
        }

    output_name = f"{sample_id}.jpg"
    original_path = ORIGINAL_DIR / output_name
    grayscale_path = GRAYSCALE_DIR / output_name
    threshold_path = THRESHOLD_DIR / output_name
    debug_path = DEBUG_DIR / output_name

    original_resized = resize_to_target_height(mrz_crop, TARGET_HEIGHT)
    grayscale = create_grayscale_variant(mrz_crop)
    thresholded = create_threshold_variant(grayscale)

    cv2.imwrite(str(original_path), original_resized)
    cv2.imwrite(str(grayscale_path), grayscale)
    cv2.imwrite(str(threshold_path), thresholded)

    debug_image = image.copy()
    bx1, by1, bx2, by2 = raw_bbox
    cv2.rectangle(debug_image, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
    confidence = parse_float(row.get("transformed_mrz_confidence"))
    label = "MRZ" if confidence is None else f"MRZ {confidence:.3f}"
    cv2.putText(
        debug_image,
        label,
        (bx1, max(25, by1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(debug_path), debug_image)

    return {
        **base,
        "status": "success_reused_stage1_detection",
        "mrz_count": 1,
        "selected_confidence": confidence,
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


def write_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row})
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = load_processing_rows()

    for directory in (
        OUTPUT_ROOT,
        ORIGINAL_DIR,
        GRAYSCALE_DIR,
        THRESHOLD_DIR,
        DEBUG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        try:
            result = process_one_row(row)
        except Exception as error:
            result = {
                "sample_id": row.get("sample_id"),
                "filename": row.get("generated_filename"),
                "source_filename": row.get("source_filename"),
                "relative_path": row.get("relative_path"),
                "status": "unexpected_error",
                "mrz_count": None,
                "selected_confidence": None,
                "error": repr(error),
            }

        output_rows.append(result)
        write_csv(output_rows)

        print(
            f"[{index:>4}/{len(rows)}] "
            f"{result.get('sample_id')} -> {result['status']}"
        )

    status_counts: dict[str, int] = {}
    for row in output_rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\n" + "=" * 72)
    print("KẾT QUẢ CROP MRZ - REUSED DETECTION")
    print("=" * 72)
    for status, count in sorted(status_counts.items()):
        print(f"{status:<40}: {count}")
    print(f"\nCSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
