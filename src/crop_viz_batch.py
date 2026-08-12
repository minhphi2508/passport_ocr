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

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "viz_stage"
COLOR_DIR = OUTPUT_ROOT / "color"
GRAYSCALE_DIR = OUTPUT_ROOT / "grayscale"
DEBUG_DIR = OUTPUT_ROOT / "debug"
CSV_PATH = OUTPUT_ROOT / "viz_crop_results.csv"

MRZ_TOP_MARGIN_RATIO = 0.04


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


def create_grayscale_variant(viz_crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(viz_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    return clahe.apply(gray)


def process_one_row(row: dict[str, str]) -> dict[str, Any]:
    sample_id = row.get("sample_id") or ""
    generated_filename = row.get("generated_filename") or f"{sample_id}.jpg"
    source_filename = row.get("source_filename") or ""
    relative_path = row.get("relative_path") or ""

    base = {
        "sample_id": sample_id,
        "filename": generated_filename,
        "source_filename": source_filename,
        "relative_path": relative_path,
    }

    transformed_path_text = row.get("transformed_path")
    if not transformed_path_text:
        return {
            **base,
            "status": "passport_page_unavailable",
            "mrz_detected": False,
            "mrz_confidence": None,
            "viz_mode": None,
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
            "mrz_detected": False,
            "mrz_confidence": None,
            "viz_mode": None,
            "error": f"Không đọc được transformed page: {image_path}",
        }

    height, width = image.shape[:2]
    mrz_detected = parse_bool(row.get("transformed_mrz_detected"))
    mrz_confidence = parse_float(row.get("transformed_mrz_confidence"))
    mrz_y1 = parse_int(row.get("transformed_mrz_y1"))

    debug_image = image.copy()

    if mrz_detected and mrz_y1 is not None:
        margin = int(round(height * MRZ_TOP_MARGIN_RATIO))
        viz_top = 0
        viz_bottom = max(1, mrz_y1 - margin)
        viz_crop = image[viz_top:viz_bottom, 0:width]
        viz_mode = "mrz_based_reused_stage1_detection"

        x1 = parse_int(row.get("transformed_mrz_x1")) or 0
        x2 = parse_int(row.get("transformed_mrz_x2")) or width - 1
        y2 = parse_int(row.get("transformed_mrz_y2")) or height - 1
        cv2.rectangle(debug_image, (x1, mrz_y1), (x2, y2), (0, 0, 255), 3)
        cv2.line(
            debug_image,
            (0, viz_bottom),
            (width - 1, viz_bottom),
            (0, 255, 0),
            3,
        )
    else:
        viz_top = 0
        viz_bottom = height
        viz_crop = image.copy()
        viz_mode = "full_page_no_mrz"
        cv2.rectangle(
            debug_image,
            (0, 0),
            (width - 1, height - 1),
            (0, 255, 255),
            4,
        )

    if viz_crop.size == 0:
        return {
            **base,
            "status": "empty_viz_crop",
            "mrz_detected": mrz_detected,
            "mrz_confidence": mrz_confidence,
            "viz_mode": viz_mode,
            "viz_top": viz_top,
            "viz_bottom": viz_bottom,
            "viz_width": 0,
            "viz_height": 0,
            "error": "VIZ crop rỗng.",
        }

    output_name = f"{sample_id}.jpg"
    color_path = COLOR_DIR / output_name
    grayscale_path = GRAYSCALE_DIR / output_name
    debug_path = DEBUG_DIR / output_name

    grayscale = create_grayscale_variant(viz_crop)

    color_success = cv2.imwrite(str(color_path), viz_crop)
    grayscale_success = cv2.imwrite(str(grayscale_path), grayscale)
    debug_success = cv2.imwrite(str(debug_path), debug_image)

    if not (color_success and grayscale_success and debug_success):
        return {
            **base,
            "status": "write_failed",
            "mrz_detected": mrz_detected,
            "mrz_confidence": mrz_confidence,
            "viz_mode": viz_mode,
            "viz_top": viz_top,
            "viz_bottom": viz_bottom,
            "viz_width": viz_crop.shape[1],
            "viz_height": viz_crop.shape[0],
            "error": "Không ghi được một hoặc nhiều output.",
        }

    return {
        **base,
        "status": "success",
        "mrz_detected": mrz_detected,
        "mrz_confidence": mrz_confidence,
        "viz_mode": viz_mode,
        "viz_top": viz_top,
        "viz_bottom": viz_bottom,
        "viz_width": viz_crop.shape[1],
        "viz_height": viz_crop.shape[0],
        "color_path": str(color_path),
        "grayscale_path": str(grayscale_path),
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

    for directory in (OUTPUT_ROOT, COLOR_DIR, GRAYSCALE_DIR, DEBUG_DIR):
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
                "mrz_detected": None,
                "mrz_confidence": None,
                "viz_mode": None,
                "error": repr(error),
            }

        output_rows.append(result)
        write_csv(output_rows)

        print(
            f"[{index:>4}/{len(rows)}] "
            f"{result.get('sample_id')} -> {result['status']} "
            f"| mode={result.get('viz_mode')}"
        )

    print(f"\nCSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
