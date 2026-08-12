from __future__ import annotations

import os

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from paddleocr import PaddleOCR

from checkpoint_utils import (
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
    build_stage_fingerprint,
    dedupe_records,
    load_jsonl,
    prepare_stage_checkpoint,
)
from device_config import PADDLE_DEVICE
from sample_manifest import sample_id_from_generated_filename


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIZ_ROOT = PROJECT_ROOT / "outputs" / "viz_stage"
VARIANT_DIRS = {
    "enhanced": VIZ_ROOT / "enhanced",
    "color": VIZ_ROOT / "color",
    "grayscale": VIZ_ROOT / "grayscale",
}

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "viz_ocr"
CSV_PATH = OUTPUT_DIR / "viz_ocr_summary.csv"
JSON_PATH = OUTPUT_DIR / "viz_ocr_full.json"
CHECKPOINT_JSONL = OUTPUT_DIR / "viz_ocr_checkpoint.jsonl"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

MIN_TEXT_COUNT = 5
MIN_MEAN_CONFIDENCE = 0.80


def result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result

    json_value = getattr(result, "json", None)
    if json_value is not None:
        if callable(json_value):
            json_value = json_value()
        if isinstance(json_value, str):
            try:
                parsed = json.loads(json_value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        if isinstance(json_value, dict):
            return json_value

    res_value = getattr(result, "res", None)
    if isinstance(res_value, dict):
        return res_value

    try:
        converted = dict(result)
        if isinstance(converted, dict):
            return converted
    except (TypeError, ValueError):
        pass

    return {}


def find_value_recursive(obj: Any, target_key: str) -> Any:
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]
        for value in obj.values():
            found = find_value_recursive(value, target_key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_value_recursive(item, target_key)
            if found is not None:
                return found
    return None


def extract_ocr_items(prediction_results: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for result in prediction_results:
        result_dict = result_to_dict(result)
        texts = find_value_recursive(result_dict, "rec_texts")
        scores = find_value_recursive(result_dict, "rec_scores")
        boxes = find_value_recursive(result_dict, "rec_boxes")
        polygons = find_value_recursive(result_dict, "rec_polys")

        if texts is None:
            continue

        texts = list(texts)
        scores = list(scores) if scores is not None else [None] * len(texts)

        try:
            boxes = list(boxes) if boxes is not None else None
        except TypeError:
            boxes = None

        try:
            polygons = list(polygons) if polygons is not None else None
        except TypeError:
            polygons = None

        for index, text in enumerate(texts):
            score = None
            if index < len(scores) and scores[index] is not None:
                try:
                    score = float(scores[index])
                except (TypeError, ValueError):
                    score = None

            box = None
            if boxes is not None and index < len(boxes):
                try:
                    box = np.asarray(boxes[index]).tolist()
                except Exception:
                    box = None

            polygon = None
            if polygons is not None and index < len(polygons):
                try:
                    polygon = np.asarray(polygons[index]).tolist()
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


def ocr_one_image(ocr: PaddleOCR, image_path: Path) -> dict[str, Any]:
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

        items = extract_ocr_items(prediction_results)
        texts = [item["text"] for item in items]
        valid_scores = [
            item["confidence"]
            for item in items
            if item["confidence"] is not None
        ]
        mean_confidence = float(np.mean(valid_scores)) if valid_scores else None

        return {
            "status": "success",
            "items": items,
            "texts": texts,
            "text_count": len(items),
            "mean_confidence": mean_confidence,
            "elapsed_seconds": time.perf_counter() - start_time,
            "error": None,
        }

    except Exception as error:
        return {
            "status": "ocr_error",
            "items": [],
            "texts": [],
            "text_count": 0,
            "mean_confidence": None,
            "elapsed_seconds": time.perf_counter() - start_time,
            "error": repr(error),
        }

    finally:
        prediction_results.clear()
        del prediction_results
        gc.collect()


def needs_fallback(result: dict[str, Any]) -> bool:
    if result["status"] != "success":
        return True
    if result["text_count"] < MIN_TEXT_COUNT:
        return True
    confidence = result["mean_confidence"]
    return confidence is None or confidence < MIN_MEAN_CONFIDENCE


def candidate_score(result: dict[str, Any]) -> float:
    if result["status"] != "success":
        return -1000.0

    score = float(result["text_count"])
    confidence = result["mean_confidence"]
    if confidence is not None:
        score += confidence * 20.0
    return score


def missing_variant(path: Path) -> dict[str, Any]:
    return {
        "status": "image_missing",
        "items": [],
        "texts": [],
        "text_count": 0,
        "mean_confidence": None,
        "elapsed_seconds": 0.0,
        "error": f"Không thấy ảnh: {path}",
    }


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    selected = record["selected_result"]
    return {
        "sample_id": record.get("sample_id"),
        "filename": record["filename"],
        "selected_variant": record["selected_variant"],
        "status": selected["status"],
        "text_count": selected["text_count"],
        "mean_confidence": selected["mean_confidence"],
        "elapsed_seconds": selected["elapsed_seconds"],
        "all_text": " | ".join(selected["texts"]),
        "fallback_used": record["fallback_used"],
        "variants_attempted": len(record["variants"]),
        "error": selected["error"],
    }


def print_summary(records: list[dict[str, Any]]) -> None:
    variant_counts: dict[str, int] = {}
    fallback_count = 0
    success_count = 0

    for record in records:
        variant = record["selected_variant"]
        variant_counts[variant] = variant_counts.get(variant, 0) + 1
        fallback_count += int(bool(record["fallback_used"]))
        success_count += int(record["selected_result"]["status"] == "success")

    print("\n" + "=" * 72)
    print("KẾT QUẢ OCR VIZ - SAFE CHECKPOINT")
    print("=" * 72)
    print(f"Tổng ảnh          : {len(records)}")
    print(f"OCR thành công    : {success_count}")
    print(f"Dùng fallback     : {fallback_count}")
    print("\nVariant được chọn:")
    for variant, count in sorted(variant_counts.items()):
        print(f"{variant:<20}: {count}")
    print(f"\nCSV : {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")


def main() -> None:
    for variant_name, directory in VARIANT_DIRS.items():
        if not directory.exists():
            raise FileNotFoundError(f"Không thấy folder {variant_name}:\n{directory}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in VARIANT_DIRS["enhanced"].rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError("Không tìm thấy enhanced VIZ.")

    fingerprint, payload = build_stage_fingerprint(
        stage_name="ocr_viz",
        code_paths=[Path(__file__), PROJECT_ROOT / "src" / "checkpoint_utils.py"],
        input_directories=VARIANT_DIRS.values(),
        input_extensions=IMAGE_EXTENSIONS,
        extra={
            "min_text_count": MIN_TEXT_COUNT,
            "min_mean_confidence": MIN_MEAN_CONFIDENCE,
        },
        packages=("paddleocr", "paddlepaddle", "paddlepaddle-gpu", "numpy"),
    )

    prepare_stage_checkpoint(
        output_dir=OUTPUT_DIR,
        stage_name="ocr_viz",
        fingerprint=fingerprint,
        fingerprint_payload=payload,
        checkpoint_paths=[CHECKPOINT_JSONL, CSV_PATH, JSON_PATH],
    )

    records = dedupe_records(load_jsonl(CHECKPOINT_JSONL), key="filename")
    completed = {str(record["filename"]) for record in records if record.get("filename")}

    print("Đang khởi tạo PaddleOCR...")
    ocr = PaddleOCR(
        lang="en",
        device=PADDLE_DEVICE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    print("PaddleOCR đã sẵn sàng.")
    print(f"Tổng VIZ              : {len(image_paths)}")
    print(f"Checkpoint compatible : {len(completed)} ảnh")
    print()

    for index, enhanced_path in enumerate(image_paths, start=1):
        filename = enhanced_path.name

        if filename in completed:
            print(
                f"[{index:>3}/{len(image_paths)}] {filename} "
                "-> skipped_checkpoint"
            )
            continue

        variant_results: dict[str, dict[str, Any]] = {}

        enhanced_result = ocr_one_image(ocr, enhanced_path)
        variant_results["enhanced"] = enhanced_result
        fallback_used = False

        if needs_fallback(enhanced_result):
            fallback_used = True
            color_path = VARIANT_DIRS["color"] / filename
            variant_results["color"] = (
                ocr_one_image(ocr, color_path)
                if color_path.exists()
                else missing_variant(color_path)
            )

        current_best_variant = max(
            variant_results,
            key=lambda name: candidate_score(variant_results[name]),
        )

        if needs_fallback(variant_results[current_best_variant]):
            fallback_used = True
            grayscale_path = VARIANT_DIRS["grayscale"] / filename
            variant_results["grayscale"] = (
                ocr_one_image(ocr, grayscale_path)
                if grayscale_path.exists()
                else missing_variant(grayscale_path)
            )

        selected_variant = max(
            variant_results,
            key=lambda name: candidate_score(variant_results[name]),
        )
        selected_result = variant_results[selected_variant]

        record = {
            "sample_id": sample_id_from_generated_filename(filename),
            "filename": filename,
            "selected_variant": selected_variant,
            "fallback_used": fallback_used,
            "selected_result": selected_result,
            "variants": variant_results,
        }

        append_jsonl(CHECKPOINT_JSONL, record)
        records.append(record)
        completed.add(filename)

        print(
            f"[{index:>3}/{len(image_paths)}] {filename} "
            f"-> {selected_variant} | texts={selected_result['text_count']} "
            f"| conf={selected_result['mean_confidence']} | fallback={fallback_used}"
        )

        gc.collect()

    records = dedupe_records(records, key="filename")
    atomic_write_json(JSON_PATH, records)
    atomic_write_csv(CSV_PATH, [flatten_record(record) for record in records])
    print_summary(records)


if __name__ == "__main__":
    main()
