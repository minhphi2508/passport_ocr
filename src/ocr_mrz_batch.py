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
from mrz_geometry import box_from_value, reconstruct_mrz_lines
from sample_manifest import sample_id_from_generated_filename
from td3_validator import validate_td3_lines


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MRZ_ROOT = PROJECT_ROOT / "outputs" / "mrz_stage"
VARIANT_DIRS = {
    "original": MRZ_ROOT / "original_crops",
    "grayscale": MRZ_ROOT / "grayscale",
    "threshold": MRZ_ROOT / "threshold",
}

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mrz_ocr"
CSV_PATH = OUTPUT_DIR / "mrz_ocr_results.csv"
JSON_PATH = OUTPUT_DIR / "mrz_ocr_results.json"
CHECKPOINT_JSONL = OUTPUT_DIR / "mrz_ocr_checkpoint.jsonl"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

VARIANT_ORDER = ("original", "grayscale", "threshold")

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
            single_text = find_value_recursive(result_dict, "rec_text")
            if single_text is not None:
                texts = [single_text]

        if texts is None:
            continue

        texts = [texts] if isinstance(texts, str) else list(texts)

        if scores is None:
            single_score = find_value_recursive(result_dict, "rec_score")
            scores = [single_score] if single_score is not None else []
        elif isinstance(scores, (float, int, np.floating, np.integer)):
            scores = [scores]
        else:
            scores = list(scores)

        try:
            boxes = list(boxes) if boxes is not None else []
        except TypeError:
            boxes = []

        try:
            polygons = list(polygons) if polygons is not None else []
        except TypeError:
            polygons = []

        for index, text in enumerate(texts):
            score = None
            if index < len(scores) and scores[index] is not None:
                try:
                    score = float(scores[index])
                except (TypeError, ValueError):
                    score = None

            box = box_from_value(boxes[index]) if index < len(boxes) else None
            if box is None and index < len(polygons):
                box = box_from_value(polygons[index])

            items.append(
                {
                    "text": str(text),
                    "confidence": score,
                    "box": box,
                }
            )

    return items


def checksum_metadata(lines: list[str]) -> dict[str, Any]:
    if len(lines) != 2 or len(lines[0]) != 44 or len(lines[1]) != 44:
        return {
            "checksum_validation_status": "not_run_invalid_structure",
            "passport_number_check_valid": None,
            "birth_date_check_valid": None,
            "expiry_date_check_valid": None,
            "personal_number_check_valid": None,
            "final_check_valid": None,
            "all_main_checks_valid": None,
            "valid_checksum_count": 0,
            "failed_checksum_count": 0,
        }

    validation = validate_td3_lines(lines[0], lines[1]).to_dict()
    check_fields = (
        "passport_number_check_valid",
        "birth_date_check_valid",
        "expiry_date_check_valid",
        "personal_number_check_valid",
        "final_check_valid",
    )

    valid_count = sum(validation.get(field) is True for field in check_fields)
    failed_count = sum(validation.get(field) is False for field in check_fields)

    return {
        "checksum_validation_status": validation.get("validation_status"),
        "passport_number_check_valid": validation.get("passport_number_check_valid"),
        "birth_date_check_valid": validation.get("birth_date_check_valid"),
        "expiry_date_check_valid": validation.get("expiry_date_check_valid"),
        "personal_number_check_valid": validation.get("personal_number_check_valid"),
        "final_check_valid": validation.get("final_check_valid"),
        "all_main_checks_valid": validation.get("all_main_checks_valid"),
        "valid_checksum_count": valid_count,
        "failed_checksum_count": failed_count,
    }


def score_candidate(
    lines: list[str],
    mean_confidence: float | None,
    checksum: dict[str, Any],
    assembly_method: str,
) -> float:
    if not lines:
        return -1000.0

    score = 0.0

    if len(lines) == 2:
        score += 100.0
    else:
        score -= abs(len(lines) - 2) * 30.0

    line_1_length = len(lines[0]) if len(lines) >= 1 else 0
    line_2_length = len(lines[1]) if len(lines) >= 2 else 0
    score -= abs(line_1_length - 44) * 3.0
    score -= abs(line_2_length - 44) * 3.0
    score -= abs(sum(len(line) for line in lines[:2]) - 88) * 0.5

    if len(lines) == 2 and line_1_length == 44 and line_2_length == 44:
        score += 120.0

    if assembly_method == "geometry_two_rows":
        score += 12.0

    if mean_confidence is not None:
        score += mean_confidence * 20.0

    score += float(checksum["valid_checksum_count"]) * 35.0
    score -= float(checksum["failed_checksum_count"]) * 12.0

    if checksum.get("final_check_valid") is True:
        score += 60.0

    if checksum.get("all_main_checks_valid") is True:
        score += 180.0

    return score


def is_strong_candidate(result: dict[str, Any]) -> bool:
    return bool(
        result.get("line_count") == 2
        and result.get("line_1_length") == 44
        and result.get("line_2_length") == 44
        and result.get("all_main_checks_valid") is True
    )


def run_ocr_on_image(ocr: PaddleOCR, image_path: Path) -> dict[str, Any]:
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
        raw_texts = [str(item.get("text") or "") for item in items]
        valid_scores = [
            float(item["confidence"])
            for item in items
            if item.get("confidence") is not None
        ]
        lines, assembly_method = reconstruct_mrz_lines(items)
        mean_confidence = float(np.mean(valid_scores)) if valid_scores else None
        checksum = checksum_metadata(lines)
        elapsed_seconds = time.perf_counter() - start_time

        return {
            "status": "success",
            "items": items,
            "raw_texts": raw_texts,
            "assembly_method": assembly_method,
            "lines": lines,
            "line_count": len(lines),
            "line_1": lines[0] if len(lines) >= 1 else None,
            "line_2": lines[1] if len(lines) >= 2 else None,
            "line_1_length": len(lines[0]) if len(lines) >= 1 else None,
            "line_2_length": len(lines[1]) if len(lines) >= 2 else None,
            "mean_confidence": mean_confidence,
            **checksum,
            "candidate_score": score_candidate(
                lines,
                mean_confidence,
                checksum,
                assembly_method,
            ),
            "strong_candidate": False,
            "elapsed_seconds": elapsed_seconds,
            "error": None,
        }

    except Exception as error:
        elapsed_seconds = time.perf_counter() - start_time
        return empty_variant_result(
            status="ocr_error",
            error=repr(error),
            elapsed_seconds=elapsed_seconds,
        )

    finally:
        prediction_results.clear()
        del prediction_results
        gc.collect()


def empty_variant_result(
    status: str,
    error: str | None,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    return {
        "status": status,
        "items": [],
        "raw_texts": [],
        "assembly_method": "none",
        "lines": [],
        "line_count": 0,
        "line_1": None,
        "line_2": None,
        "line_1_length": None,
        "line_2_length": None,
        "mean_confidence": None,
        "checksum_validation_status": "not_run",
        "passport_number_check_valid": None,
        "birth_date_check_valid": None,
        "expiry_date_check_valid": None,
        "personal_number_check_valid": None,
        "final_check_valid": None,
        "all_main_checks_valid": None,
        "valid_checksum_count": 0,
        "failed_checksum_count": 0,
        "candidate_score": -1000.0,
        "strong_candidate": False,
        "elapsed_seconds": elapsed_seconds,
        "error": error,
    }


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    selected = record["selected_result"]
    row: dict[str, Any] = {
        "sample_id": record.get("sample_id"),
        "filename": record["filename"],
        "selected_variant": record["selected_variant"],
        "adaptive_stopped_early": record["adaptive_stopped_early"],
        "variants_attempted": len(record["variants"]),
        "selected_status": selected["status"],
        "selected_assembly_method": selected.get("assembly_method"),
        "selected_line_count": selected["line_count"],
        "selected_line_1": selected["line_1"],
        "selected_line_2": selected["line_2"],
        "selected_line_1_length": selected["line_1_length"],
        "selected_line_2_length": selected["line_2_length"],
        "selected_mean_confidence": selected["mean_confidence"],
        "selected_candidate_score": selected["candidate_score"],
        "selected_valid_checksum_count": selected["valid_checksum_count"],
        "selected_failed_checksum_count": selected["failed_checksum_count"],
        "selected_all_main_checks_valid": selected["all_main_checks_valid"],
        "selected_final_check_valid": selected["final_check_valid"],
    }

    for variant_name in VARIANT_ORDER:
        result = record["variants"].get(variant_name)
        prefix = f"{variant_name}_"

        if result is None:
            row[f"{prefix}attempted"] = False
            continue

        row[f"{prefix}attempted"] = True
        for key in (
            "status",
            "assembly_method",
            "line_count",
            "line_1",
            "line_2",
            "line_1_length",
            "line_2_length",
            "mean_confidence",
            "candidate_score",
            "valid_checksum_count",
            "failed_checksum_count",
            "all_main_checks_valid",
            "final_check_valid",
            "elapsed_seconds",
            "error",
        ):
            row[f"{prefix}{key}"] = result.get(key)

    return row


def print_final_summary(records: list[dict[str, Any]]) -> None:
    csv_rows = [flatten_record(record) for record in records]
    exact_44_44 = sum(
        row.get("selected_line_1_length") == 44
        and row.get("selected_line_2_length") == 44
        for row in csv_rows
    )
    checksum_valid = sum(
        row.get("selected_all_main_checks_valid") is True for row in csv_rows
    )
    early_stop = sum(bool(row.get("adaptive_stopped_early")) for row in csv_rows)
    geometry_used = sum(
        row.get("selected_assembly_method") == "geometry_two_rows"
        for row in csv_rows
    )

    print("\n" + "=" * 76)
    print("KẾT QUẢ OCR MRZ - GEOMETRY / CHECKSUM / ADAPTIVE")
    print("=" * 76)
    print(f"Tổng ảnh OCR                  : {len(records)}")
    print(f"Đúng chính xác 44 + 44        : {exact_44_44}")
    print(f"All main checks valid         : {checksum_valid}")
    print(f"Geometry reconstruction       : {geometry_used}")
    print(f"Dừng sớm trước đủ 3 variants  : {early_stop}")
    print(f"\nCSV : {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")


def main() -> None:
    for variant_name, directory in VARIANT_DIRS.items():
        if not directory.exists():
            raise FileNotFoundError(f"Không thấy thư mục {variant_name}:\n{directory}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_images = sorted(
        path
        for path in VARIANT_DIRS["original"].rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not original_images:
        raise RuntimeError("Không tìm thấy ảnh MRZ để OCR.")

    fingerprint, payload = build_stage_fingerprint(
        stage_name="ocr_mrz",
        code_paths=[
            Path(__file__),
            PROJECT_ROOT / "src" / "td3_validator.py",
            PROJECT_ROOT / "src" / "mrz_geometry.py",
            PROJECT_ROOT / "src" / "checkpoint_utils.py",
        ],
        input_directories=VARIANT_DIRS.values(),
        input_extensions=IMAGE_EXTENSIONS,
        extra={"variant_order": VARIANT_ORDER},
        packages=("paddleocr", "paddlepaddle", "paddlepaddle-gpu", "numpy"),
    )

    prepare_stage_checkpoint(
        output_dir=OUTPUT_DIR,
        stage_name="ocr_mrz",
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
    print(f"Tổng ảnh MRZ          : {len(original_images)}")
    print(f"Checkpoint compatible : {len(completed)} ảnh")
    print()

    for index, original_path in enumerate(original_images, start=1):
        filename = original_path.name

        if filename in completed:
            print(
                f"[{index:>3}/{len(original_images)}] {filename} "
                "-> skipped_checkpoint"
            )
            continue

        variant_results: dict[str, dict[str, Any]] = {}
        adaptive_stopped_early = False

        for variant_index, variant_name in enumerate(VARIANT_ORDER):
            variant_path = VARIANT_DIRS[variant_name] / filename

            if not variant_path.exists():
                result = empty_variant_result(
                    status="image_missing",
                    error=f"Không thấy ảnh: {variant_path}",
                )
            else:
                result = run_ocr_on_image(ocr, variant_path)

            result["strong_candidate"] = is_strong_candidate(result)
            variant_results[variant_name] = result

            if result["strong_candidate"]:
                adaptive_stopped_early = variant_index < len(VARIANT_ORDER) - 1
                break

            gc.collect()

        selected_variant = max(
            variant_results,
            key=lambda name: variant_results[name]["candidate_score"],
        )
        selected_result = variant_results[selected_variant]

        record = {
            "sample_id": sample_id_from_generated_filename(filename),
            "filename": filename,
            "selected_variant": selected_variant,
            "adaptive_stopped_early": adaptive_stopped_early,
            "selected_result": selected_result,
            "variants": variant_results,
        }

        append_jsonl(CHECKPOINT_JSONL, record)
        records.append(record)
        completed.add(filename)

        print(
            f"[{index:>3}/{len(original_images)}] {filename} "
            f"-> {selected_variant} | "
            f"assembly={selected_result['assembly_method']} | "
            f"attempted={len(variant_results)} | "
            f"44+44={selected_result['line_1_length'] == 44 and selected_result['line_2_length'] == 44} | "
            f"checks={selected_result['valid_checksum_count']} | "
            f"all_main={selected_result['all_main_checks_valid']}"
        )

        gc.collect()

    records = dedupe_records(records, key="filename")
    atomic_write_json(JSON_PATH, records)
    atomic_write_csv(CSV_PATH, [flatten_record(record) for record in records])
    print_final_summary(records)


if __name__ == "__main__":
    main()
