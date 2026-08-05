from __future__ import annotations

# ============================================================
# BIẾN MÔI TRƯỜNG — PHẢI ĐẶT TRƯỚC KHI IMPORT PADDLEOCR
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
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from paddleocr import PaddleOCR
from device_config import PADDLE_DEVICE


# ============================================================
# CẤU HÌNH
# ============================================================

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
# CHUẨN HÓA TEXT MRZ
# ============================================================

def normalize_mrz_piece(text: str) -> str:
    """
    Chỉ làm sạch định dạng cơ bản.

    Chưa tự sửa O/0, I/1, B/8 vì việc đó sẽ được thực hiện
    sau khi parse trường và kiểm tra check digit.
    """
    text = str(text).upper().strip()

    text = text.replace("«", "<")
    text = text.replace("‹", "<")
    text = text.replace("＜", "<")
    text = text.replace(" ", "")

    # Chỉ giữ ký tự hợp lệ trong MRZ.
    text = re.sub(r"[^A-Z0-9<]", "", text)

    return text


def merge_fragments_into_lines(
    fragments: list[str],
) -> list[str]:
    """
    PaddleOCR đôi khi:
    - trả đúng hai dòng;
    - tách một dòng thành nhiều fragment;
    - ghép cả hai dòng thành một chuỗi.

    Hàm này cố gắng đưa kết quả về tối đa hai dòng MRZ.
    """
    cleaned = [
        normalize_mrz_piece(fragment)
        for fragment in fragments
    ]

    cleaned = [
        fragment
        for fragment in cleaned
        if fragment
    ]

    if not cleaned:
        return []

    # Trường hợp lý tưởng.
    if len(cleaned) == 2:
        return cleaned

    # OCR ghép cả hai dòng thành một chuỗi.
    if len(cleaned) == 1:
        text = cleaned[0]

        if len(text) >= 80:
            # Chọn điểm chia để hai dòng gần 44 ký tự nhất.
            best_split = min(
                range(1, len(text)),
                key=lambda index: (
                    abs(index - 44)
                    + abs((len(text) - index) - 44)
                ),
            )

            return [
                text[:best_split],
                text[best_split:],
            ]

        return cleaned

    # OCR tách thành nhiều fragment.
    total_text = "".join(cleaned)

    if len(total_text) >= 80:
        best_split = min(
            range(1, len(total_text)),
            key=lambda index: (
                abs(index - 44)
                + abs((len(total_text) - index) - 44)
            ),
        )

        return [
            total_text[:best_split],
            total_text[best_split:],
        ]

    return cleaned


# ============================================================
# ĐỌC OUTPUT PADDLEOCR 3.X
# ============================================================

def result_to_dict(result: Any) -> dict[str, Any]:
    """
    Chuyển Result object của PaddleOCR/PaddleX về dictionary.
    """
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


def find_value_recursive(
    obj: Any,
    target_key: str,
) -> Any:
    """
    Tìm key trong dictionary/list lồng nhau.
    """
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


def extract_texts_and_scores(
    prediction_results: list[Any],
) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    scores: list[float] = []

    for result in prediction_results:
        result_dict = result_to_dict(result)

        rec_texts = find_value_recursive(
            result_dict,
            "rec_texts",
        )

        rec_scores = find_value_recursive(
            result_dict,
            "rec_scores",
        )

        # Fallback cho output dạng một text duy nhất.
        if rec_texts is None:
            single_text = find_value_recursive(
                result_dict,
                "rec_text",
            )

            if single_text is not None:
                rec_texts = [single_text]

        if rec_scores is None:
            single_score = find_value_recursive(
                result_dict,
                "rec_score",
            )

            if single_score is not None:
                rec_scores = [single_score]

        if rec_texts is not None:
            if isinstance(rec_texts, str):
                texts.append(rec_texts)
            else:
                texts.extend(
                    str(text)
                    for text in rec_texts
                )

        if rec_scores is not None:
            if isinstance(
                rec_scores,
                (
                    float,
                    int,
                    np.floating,
                    np.integer,
                ),
            ):
                scores.append(float(rec_scores))

            else:
                for score in rec_scores:
                    try:
                        scores.append(float(score))
                    except (TypeError, ValueError):
                        continue

    return texts, scores


# ============================================================
# CHẤM ĐIỂM KẾT QUẢ OCR
# ============================================================

def score_candidate(
    lines: list[str],
    mean_confidence: float | None,
) -> float:
    """
    Checksum chưa được dùng ở bước này.

    Ưu tiên:
    1. đúng hai dòng;
    2. mỗi dòng gần 44 ký tự;
    3. confidence cao.
    """
    if not lines:
        return -1000.0

    score = 0.0

    if len(lines) == 2:
        score += 100.0
    else:
        score -= abs(len(lines) - 2) * 30.0

    line_1_length = (
        len(lines[0])
        if len(lines) >= 1
        else 0
    )

    line_2_length = (
        len(lines[1])
        if len(lines) >= 2
        else 0
    )

    score -= abs(line_1_length - 44) * 3.0
    score -= abs(line_2_length - 44) * 3.0

    total_length = sum(
        len(line)
        for line in lines[:2]
    )

    score -= abs(total_length - 88) * 0.5

    if mean_confidence is not None:
        score += mean_confidence * 20.0

    return score


# ============================================================
# OCR MỘT ẢNH
# ============================================================

def run_ocr_on_image(
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

        raw_texts, scores = extract_texts_and_scores(
            prediction_results
        )

        lines = merge_fragments_into_lines(
            raw_texts
        )

        mean_confidence = (
            float(np.mean(scores))
            if scores
            else None
        )

        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        result = {
            "status": "success",
            "raw_texts": raw_texts,
            "lines": lines,
            "line_count": len(lines),
            "line_1": (
                lines[0]
                if len(lines) >= 1
                else None
            ),
            "line_2": (
                lines[1]
                if len(lines) >= 2
                else None
            ),
            "line_1_length": (
                len(lines[0])
                if len(lines) >= 1
                else None
            ),
            "line_2_length": (
                len(lines[1])
                if len(lines) >= 2
                else None
            ),
            "mean_confidence": mean_confidence,
            "candidate_score": score_candidate(
                lines,
                mean_confidence,
            ),
            "elapsed_seconds": elapsed_seconds,
            "error": None,
        }

    except Exception as error:
        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        result = {
            "status": "ocr_error",
            "raw_texts": [],
            "lines": [],
            "line_count": 0,
            "line_1": None,
            "line_2": None,
            "line_1_length": None,
            "line_2_length": None,
            "mean_confidence": None,
            "candidate_score": -1000.0,
            "elapsed_seconds": elapsed_seconds,
            "error": repr(error),
        }

    finally:
        prediction_results.clear()
        del prediction_results
        gc.collect()

    return result


# ============================================================
# CHUYỂN KẾT QUẢ THÀNH MỘT DÒNG CSV
# ============================================================

def flatten_row(
    filename: str,
    variant_results: dict[str, dict[str, Any]],
    selected_variant: str,
) -> dict[str, Any]:
    selected = variant_results[selected_variant]

    row: dict[str, Any] = {
        "filename": filename,
        "selected_variant": selected_variant,
        "selected_status": selected["status"],
        "selected_line_count": selected["line_count"],
        "selected_line_1": selected["line_1"],
        "selected_line_2": selected["line_2"],
        "selected_line_1_length": selected["line_1_length"],
        "selected_line_2_length": selected["line_2_length"],
        "selected_mean_confidence": selected["mean_confidence"],
        "selected_candidate_score": selected["candidate_score"],
    }

    for variant_name, result in variant_results.items():
        prefix = f"{variant_name}_"

        row[f"{prefix}status"] = result["status"]
        row[f"{prefix}line_count"] = result["line_count"]
        row[f"{prefix}line_1"] = result["line_1"]
        row[f"{prefix}line_2"] = result["line_2"]
        row[f"{prefix}line_1_length"] = result["line_1_length"]
        row[f"{prefix}line_2_length"] = result["line_2_length"]
        row[f"{prefix}mean_confidence"] = result["mean_confidence"]
        row[f"{prefix}candidate_score"] = result["candidate_score"]
        row[f"{prefix}elapsed_seconds"] = result["elapsed_seconds"]
        row[f"{prefix}error"] = result["error"]

    return row


# ============================================================
# CHECKPOINT
# ============================================================

def load_existing_csv_rows() -> list[dict[str, Any]]:
    if not CSV_PATH.exists():
        return []

    try:
        with CSV_PATH.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            return list(csv.DictReader(csv_file))

    except (OSError, csv.Error):
        return []


def load_existing_json_records() -> list[dict[str, Any]]:
    if not JSON_PATH.exists():
        return []

    try:
        with JSON_PATH.open(
            "r",
            encoding="utf-8",
        ) as json_file:
            records = json.load(json_file)

        if isinstance(records, list):
            return records

    except (OSError, json.JSONDecodeError):
        pass

    return []


def write_csv(
    rows: list[dict[str, Any]],
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


def write_json(
    records: list[dict[str, Any]],
) -> None:
    with JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            records,
            json_file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# THỐNG KÊ
# ============================================================

def print_final_summary(
    csv_rows: list[dict[str, Any]],
) -> None:
    exact_two_lines = sum(
        1
        for row in csv_rows
        if str(row.get("selected_line_count")) == "2"
    )

    exact_44_44 = sum(
        1
        for row in csv_rows
        if (
            str(row.get("selected_line_1_length")) == "44"
            and str(row.get("selected_line_2_length")) == "44"
        )
    )

    selected_variant_counts: dict[str, int] = {}

    for row in csv_rows:
        variant = str(
            row.get("selected_variant", "unknown")
        )

        selected_variant_counts[variant] = (
            selected_variant_counts.get(variant, 0)
            + 1
        )

    print("\n" + "=" * 76)
    print("KẾT QUẢ OCR MRZ")
    print("=" * 76)

    print(f"Tổng ảnh OCR               : {len(csv_rows)}")
    print(f"Đúng 2 dòng                : {exact_two_lines}")
    print(f"Đúng chính xác 44 + 44     : {exact_44_44}")

    print("\nBiến thể được chọn:")

    for variant, count in sorted(
        selected_variant_counts.items()
    ):
        print(f"{variant:<20}: {count}")

    print("\nCSV:")
    print(CSV_PATH)

    print("\nJSON:")
    print(JSON_PATH)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    for variant_name, directory in VARIANT_DIRS.items():
        if not directory.exists():
            raise FileNotFoundError(
                f"Không thấy thư mục {variant_name}:\n"
                f"{directory}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_images = sorted(
        path
        for path in VARIANT_DIRS["original"].rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )

    if not original_images:
        raise RuntimeError(
            "Không tìm thấy ảnh MRZ để OCR."
        )

    csv_rows = load_existing_csv_rows()
    json_records = load_existing_json_records()

    completed_filenames = {
        str(row["filename"])
        for row in csv_rows
        if row.get("filename")
    }

    print("Đang khởi tạo PaddleOCR...")

    ocr = PaddleOCR(
    lang="en",
    device=PADDLE_DEVICE,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

    print("PaddleOCR đã sẵn sàng.")
    print(f"Tổng ảnh MRZ              : {len(original_images)}")
    print(f"Đã hoàn thành trước đó    : {len(completed_filenames)}")
    print(
        f"Còn lại                   : "
        f"{len(original_images) - len(completed_filenames)}"
    )
    print("Mỗi ảnh sẽ chạy 3 biến thể.")
    print()

    for index, original_path in enumerate(
        original_images,
        start=1,
    ):
        filename = original_path.name

        if filename in completed_filenames:
            print(
                f"[{index:>3}/{len(original_images)}] "
                f"{filename} "
                f"-> skipped_already_completed"
            )
            continue

        variant_results: dict[str, dict[str, Any]] = {}

        for variant_name, directory in VARIANT_DIRS.items():
            variant_path = directory / filename

            if not variant_path.exists():
                variant_results[variant_name] = {
                    "status": "image_missing",
                    "raw_texts": [],
                    "lines": [],
                    "line_count": 0,
                    "line_1": None,
                    "line_2": None,
                    "line_1_length": None,
                    "line_2_length": None,
                    "mean_confidence": None,
                    "candidate_score": -1000.0,
                    "elapsed_seconds": 0.0,
                    "error": (
                        f"Không thấy ảnh: {variant_path}"
                    ),
                }

            else:
                variant_results[variant_name] = (
                    run_ocr_on_image(
                        ocr=ocr,
                        image_path=variant_path,
                    )
                )

            gc.collect()

        selected_variant = max(
            variant_results,
            key=lambda name: variant_results[name][
                "candidate_score"
            ],
        )

        selected_result = variant_results[
            selected_variant
        ]

        csv_row = flatten_row(
            filename=filename,
            variant_results=variant_results,
            selected_variant=selected_variant,
        )

        json_record = {
            "filename": filename,
            "selected_variant": selected_variant,
            "selected_result": selected_result,
            "variants": variant_results,
        }

        csv_rows.append(csv_row)
        json_records.append(json_record)
        completed_filenames.add(filename)

        # Checkpoint sau từng ảnh.
        write_csv(csv_rows)
        write_json(json_records)

        print(
            f"[{index:>3}/{len(original_images)}] "
            f"{filename} "
            f"-> {selected_variant} | "
            f"lines={selected_result['line_count']} | "
            f"lengths="
            f"{selected_result['line_1_length']},"
            f"{selected_result['line_2_length']} | "
            f"conf={selected_result['mean_confidence']}"
        )

        del variant_results
        del selected_result
        del csv_row
        del json_record

        gc.collect()

    print_final_summary(csv_rows)


if __name__ == "__main__":
    main()