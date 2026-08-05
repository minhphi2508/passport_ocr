from __future__ import annotations

# ============================================================
# ENVIRONMENT — phải đặt trước khi import PaddleOCR
# ============================================================

import os

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


# ============================================================
# IMPORT
# ============================================================

import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from paddleocr import PaddleOCR
from device_config import PADDLE_DEVICE


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIZ_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "viz_stage"
)

VARIANT_DIRS = {
    "enhanced": VIZ_ROOT / "enhanced",
    "color": VIZ_ROOT / "color",
    "grayscale": VIZ_ROOT / "grayscale",
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "viz_ocr"
)

CSV_PATH = OUTPUT_DIR / "viz_ocr_summary.csv"
JSON_PATH = OUTPUT_DIR / "viz_ocr_full.json"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

# ============================================================
# NGƯỠNG FALLBACK
# ============================================================

MIN_TEXT_COUNT = 5
MIN_MEAN_CONFIDENCE = 0.80


# ============================================================
# RESULT HELPERS
# ============================================================

def result_to_dict(
    result: Any,
) -> dict[str, Any]:

    if isinstance(result, dict):
        return result

    json_value = getattr(
        result,
        "json",
        None,
    )

    if json_value is not None:

        if callable(json_value):
            json_value = json_value()

        if isinstance(json_value, str):

            try:
                parsed = json.loads(
                    json_value
                )

                if isinstance(parsed, dict):
                    return parsed

            except json.JSONDecodeError:
                pass

        if isinstance(json_value, dict):
            return json_value

    res_value = getattr(
        result,
        "res",
        None,
    )

    if isinstance(res_value, dict):
        return res_value

    try:
        converted = dict(result)

        if isinstance(converted, dict):
            return converted

    except (TypeError, ValueError):
        pass

    return {}


def find_value_recursive(
    obj: Any,
    target_key: str,
) -> Any:

    if isinstance(obj, dict):

        if target_key in obj:
            return obj[target_key]

        for value in obj.values():

            found = find_value_recursive(
                value,
                target_key,
            )

            if found is not None:
                return found

    elif isinstance(obj, list):

        for item in obj:

            found = find_value_recursive(
                item,
                target_key,
            )

            if found is not None:
                return found

    return None


# ============================================================
# EXTRACT OCR ITEMS
# ============================================================

def extract_ocr_items(
    prediction_results: list[Any],
) -> list[dict[str, Any]]:

    items: list[dict[str, Any]] = []

    for result in prediction_results:

        result_dict = result_to_dict(
            result
        )

        texts = find_value_recursive(
            result_dict,
            "rec_texts",
        )

        scores = find_value_recursive(
            result_dict,
            "rec_scores",
        )

        boxes = find_value_recursive(
            result_dict,
            "rec_boxes",
        )

        polygons = find_value_recursive(
            result_dict,
            "rec_polys",
        )

        if texts is None:
            continue

        texts = list(texts)

        if scores is None:
            scores = [None] * len(texts)
        else:
            scores = list(scores)

        if boxes is not None:
            try:
                boxes = list(boxes)
            except TypeError:
                boxes = None

        if polygons is not None:
            try:
                polygons = list(polygons)
            except TypeError:
                polygons = None

        for index, text in enumerate(texts):

            score = None

            if (
                index < len(scores)
                and scores[index] is not None
            ):
                try:
                    score = float(
                        scores[index]
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    score = None

            box = None

            if (
                boxes is not None
                and index < len(boxes)
            ):
                try:
                    box = (
                        np.asarray(
                            boxes[index]
                        )
                        .tolist()
                    )
                except Exception:
                    box = None

            polygon = None

            if (
                polygons is not None
                and index < len(polygons)
            ):
                try:
                    polygon = (
                        np.asarray(
                            polygons[index]
                        )
                        .tolist()
                    )
                except Exception:
                    polygon = None

            items.append(
                {
                    "text": str(text),
                    "confidence": score,
                    "box": box,
                    "polygon": polygon,
                }
            )

    return items


# ============================================================
# OCR ONE VARIANT
# ============================================================

def ocr_one_image(
    ocr: PaddleOCR,
    image_path: Path,
) -> dict[str, Any]:

    start_time = time.perf_counter()

    prediction_results: list[Any] = []

    try:

        prediction_results = list(
            ocr.predict(
                input=str(image_path),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        )

        items = extract_ocr_items(
            prediction_results
        )

        texts = [
            item["text"]
            for item in items
        ]

        valid_scores = [
            item["confidence"]
            for item in items
            if item["confidence"] is not None
        ]

        mean_confidence = (
            float(
                np.mean(valid_scores)
            )
            if valid_scores
            else None
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "status": "success",
            "items": items,
            "texts": texts,
            "text_count": len(items),
            "mean_confidence": (
                mean_confidence
            ),
            "elapsed_seconds": elapsed,
            "error": None,
        }

    except Exception as error:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "status": "ocr_error",
            "items": [],
            "texts": [],
            "text_count": 0,
            "mean_confidence": None,
            "elapsed_seconds": elapsed,
            "error": repr(error),
        }

    finally:

        prediction_results.clear()
        del prediction_results

        gc.collect()


# ============================================================
# QUALITY / FALLBACK
# ============================================================

def needs_fallback(
    result: dict[str, Any],
) -> bool:

    if result["status"] != "success":
        return True

    if result["text_count"] < MIN_TEXT_COUNT:
        return True

    confidence = result[
        "mean_confidence"
    ]

    if confidence is None:
        return True

    if confidence < MIN_MEAN_CONFIDENCE:
        return True

    return False


def candidate_score(
    result: dict[str, Any],
) -> float:

    if result["status"] != "success":
        return -1000.0

    score = 0.0

    score += result[
        "text_count"
    ] * 1.0

    confidence = result[
        "mean_confidence"
    ]

    if confidence is not None:
        score += (
            confidence * 20.0
        )

    return score


# ============================================================
# CHECKPOINT
# ============================================================

def load_existing_records(
) -> list[dict[str, Any]]:

    if not JSON_PATH.exists():
        return []

    try:

        with JSON_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if isinstance(data, list):
            return data

    except (
        OSError,
        json.JSONDecodeError,
    ):
        pass

    return []


def write_json(
    records: list[dict[str, Any]],
) -> None:

    with JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )


def write_csv(
    records: list[dict[str, Any]],
) -> None:

    rows = []

    for record in records:

        selected = record[
            "selected_result"
        ]

        rows.append(
            {
                "filename": record[
                    "filename"
                ],
                "selected_variant": (
                    record[
                        "selected_variant"
                    ]
                ),
                "status": selected[
                    "status"
                ],
                "text_count": selected[
                    "text_count"
                ],
                "mean_confidence": (
                    selected[
                        "mean_confidence"
                    ]
                ),
                "elapsed_seconds": (
                    selected[
                        "elapsed_seconds"
                    ]
                ),
                "all_text": " | ".join(
                    selected["texts"]
                ),
                "fallback_used": (
                    record[
                        "fallback_used"
                    ]
                ),
                "error": selected[
                    "error"
                ],
            }
        )

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    for (
        variant_name,
        directory,
    ) in VARIANT_DIRS.items():

        if not directory.exists():
            raise FileNotFoundError(
                f"Không thấy folder "
                f"{variant_name}:\n"
                f"{directory}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        path
        for path in (
            VARIANT_DIRS[
                "enhanced"
            ]
        ).rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            "Không tìm thấy enhanced VIZ."
        )

    records = (
        load_existing_records()
    )

    completed = {
        record["filename"]
        for record in records
        if record.get(
            "filename"
        )
    }

    print(
        "Đang khởi tạo PaddleOCR..."
    )

    ocr = PaddleOCR(
        lang="en",
        device=PADDLE_DEVICE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print(
        "PaddleOCR đã sẵn sàng."
    )

    print(
        f"Tổng VIZ          : "
        f"{len(image_paths)}"
    )

    print(
        f"Đã hoàn thành     : "
        f"{len(completed)}"
    )

    print(
        f"Còn lại           : "
        f"{len(image_paths) - len(completed)}"
    )

    print()

    for index, enhanced_path in enumerate(
        image_paths,
        start=1,
    ):

        filename = (
            enhanced_path.name
        )

        if filename in completed:

            print(
                f"[{index:>3}/"
                f"{len(image_paths)}] "
                f"{filename} "
                f"-> skipped"
            )

            continue

        variant_results = {}

        # ====================================================
        # 1. ENHANCED
        # ====================================================

        enhanced_result = (
            ocr_one_image(
                ocr=ocr,
                image_path=enhanced_path,
            )
        )

        variant_results[
            "enhanced"
        ] = enhanced_result

        fallback_used = False

        # ====================================================
        # 2. FALLBACK COLOR
        # ====================================================

        if needs_fallback(
            enhanced_result
        ):

            fallback_used = True

            color_path = (
                VARIANT_DIRS[
                    "color"
                ]
                / filename
            )

            color_result = (
                ocr_one_image(
                    ocr=ocr,
                    image_path=color_path,
                )
            )

            variant_results[
                "color"
            ] = color_result

        # ====================================================
        # 3. FALLBACK GRAYSCALE
        # ====================================================

        current_best_variant = max(
            variant_results,
            key=lambda name: (
                candidate_score(
                    variant_results[
                        name
                    ]
                )
            ),
        )

        current_best = (
            variant_results[
                current_best_variant
            ]
        )

        if needs_fallback(
            current_best
        ):

            fallback_used = True

            grayscale_path = (
                VARIANT_DIRS[
                    "grayscale"
                ]
                / filename
            )

            grayscale_result = (
                ocr_one_image(
                    ocr=ocr,
                    image_path=grayscale_path,
                )
            )

            variant_results[
                "grayscale"
            ] = grayscale_result

        # ====================================================
        # CHỌN BEST
        # ====================================================

        selected_variant = max(
            variant_results,
            key=lambda name: (
                candidate_score(
                    variant_results[
                        name
                    ]
                )
            ),
        )

        selected_result = (
            variant_results[
                selected_variant
            ]
        )

        record = {
            "filename": filename,
            "selected_variant": (
                selected_variant
            ),
            "fallback_used": (
                fallback_used
            ),
            "selected_result": (
                selected_result
            ),
            "variants": (
                variant_results
            ),
        }

        records.append(
            record
        )

        completed.add(
            filename
        )

        write_json(
            records
        )

        write_csv(
            records
        )

        print(
            f"[{index:>3}/"
            f"{len(image_paths)}] "
            f"{filename} "
            f"-> {selected_variant} "
            f"| texts="
            f"{selected_result['text_count']} "
            f"| conf="
            f"{selected_result['mean_confidence']} "
            f"| fallback="
            f"{fallback_used}"
        )

        del enhanced_result
        del selected_result
        del variant_results
        del record

        gc.collect()

    # ========================================================
    # SUMMARY
    # ========================================================

    variant_counts = {}

    fallback_count = 0
    success_count = 0

    for record in records:

        variant = record[
            "selected_variant"
        ]

        variant_counts[
            variant
        ] = (
            variant_counts.get(
                variant,
                0,
            )
            + 1
        )

        if record[
            "fallback_used"
        ]:
            fallback_count += 1

        if (
            record[
                "selected_result"
            ][
                "status"
            ]
            == "success"
        ):
            success_count += 1

    print(
        "\n"
        + "=" * 72
    )

    print(
        "KẾT QUẢ OCR VIZ"
    )

    print(
        "=" * 72
    )

    print(
        f"Tổng ảnh          : "
        f"{len(records)}"
    )

    print(
        f"OCR thành công    : "
        f"{success_count}"
    )

    print(
        f"Dùng fallback     : "
        f"{fallback_count}"
    )

    print(
        "\nVariant được chọn:"
    )

    for (
        variant,
        count,
    ) in sorted(
        variant_counts.items()
    ):

        print(
            f"{variant:<20}: "
            f"{count}"
        )

    print(
        "\nCSV:"
    )

    print(
        CSV_PATH
    )

    print(
        "\nJSON:"
    )

    print(
        JSON_PATH
    )


if __name__ == "__main__":
    main()