from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from device_config import print_device_summary


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

INPUT_DIR = PROJECT_ROOT / "input_images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# PIPELINE STAGES
# ============================================================

STAGES = [
    {
        "name": "Detect + Process Passport Pages",
        "script": "process_passport_pages.py",
    },
    {
        "name": "Crop MRZ",
        "script": "crop_mrz_batch.py",
    },
    {
        "name": "OCR MRZ",
        "script": "ocr_mrz_batch.py",
    },
    {
        "name": "Parse TD3",
        "script": "parse_mrz_results.py",
    },
    {
        "name": "Validate MRZ",
        "script": "validate_mrz_results.py",
    },
    {
        "name": "Crop VIZ",
        "script": "crop_viz_batch.py",
    },
    {
        "name": "Preprocess VIZ",
        "script": "preprocess_viz_batch.py",
    },
    {
        "name": "OCR VIZ",
        "script": "ocr_viz_batch.py",
    },
    {
        "name": "Extract Date of Issue",
        "script": "extract_date_of_issue.py",
    },
    {
        "name": "Build Final Results",
        "script": "build_final_results.py",
    },
]


# ============================================================
# OUTPUT FOLDERS
# ============================================================

#
# Chỉ các output được sinh tự động bởi pipeline.
# Không đụng vào:
# - input_images
# - models
# - src
#
GENERATED_OUTPUT_DIRS = [
    OUTPUTS_DIR / "passport_pages_safe",
    OUTPUTS_DIR / "mrz_stage",
    OUTPUTS_DIR / "mrz_ocr",
    OUTPUTS_DIR / "mrz_parsed",
    OUTPUTS_DIR / "mrz_validated",
    OUTPUTS_DIR / "viz_stage",
    OUTPUTS_DIR / "viz_ocr",
    OUTPUTS_DIR / "date_of_issue_hybrid_v3",
    OUTPUTS_DIR / "final_results",
]


# ============================================================
# VALIDATION
# ============================================================

def check_scripts() -> None:
    missing = []

    for stage in STAGES:
        script_path = SRC_DIR / stage["script"]

        if not script_path.exists():
            missing.append(script_path)

    if missing:
        print("\nThiếu script:")

        for path in missing:
            print(f"  - {path}")

        raise FileNotFoundError(
            "Không thể chạy pipeline vì thiếu script."
        )


def check_input_images() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Không thấy input folder:\n{INPUT_DIR}"
        )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }

    images = [
        path
        for path in INPUT_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in image_extensions
        )
    ]

    if not images:
        raise RuntimeError(
            f"Không có ảnh trong:\n{INPUT_DIR}"
        )

    print(
        f"Input images : {len(images)}"
    )


# ============================================================
# CLEAN OUTPUT
# ============================================================

def clean_generated_outputs() -> None:
    print()
    print("=" * 76)
    print("FRESH RUN - XÓA OUTPUT CŨ")
    print("=" * 76)

    for directory in GENERATED_OUTPUT_DIRS:
        if directory.exists():
            shutil.rmtree(directory)

            print(
                f"Removed: "
                f"{directory.relative_to(PROJECT_ROOT)}"
            )

    print()
    print(
        "✓ Output cũ đã được dọn."
    )


# ============================================================
# RUN ONE STAGE
# ============================================================

def run_stage(
    stage_number: int,
    total_stages: int,
    name: str,
    script: str,
) -> None:

    script_path = SRC_DIR / script

    print()
    print("=" * 76)

    print(
        f"[{stage_number}/{total_stages}] "
        f"{name}"
    )

    print("=" * 76)

    print(
        f"Script: {script}"
    )

    start_time = time.time()

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
    )

    elapsed = (
        time.time()
        - start_time
    )

    if result.returncode != 0:
        print()
        print("=" * 76)
        print("PIPELINE DỪNG")
        print("=" * 76)

        print(
            f"Stage lỗi : {name}"
        )

        print(
            f"Script     : {script}"
        )

        print(
            f"Return code: "
            f"{result.returncode}"
        )

        raise RuntimeError(
            f"Stage failed: {name}"
        )

    print()
    print(
        f"✓ Hoàn thành: {name}"
    )

    print(
        f"Thời gian   : "
        f"{elapsed:.1f}s"
    )


# ============================================================
# FINAL OUTPUT CHECK
# ============================================================

def print_final_output() -> None:

    final_csv = (
        OUTPUTS_DIR
        / "final_results"
        / "passport_extraction_results.csv"
    )

    final_json = (
        OUTPUTS_DIR
        / "final_results"
        / "passport_extraction_results.json"
    )

    print()
    print("=" * 76)
    print("FINAL OUTPUT")
    print("=" * 76)

    if final_csv.exists():
        print(
            f"CSV : {final_csv}"
        )

    else:
        print(
            "CSV : chưa được tạo"
        )

    if final_json.exists():
        print(
            f"JSON: {final_json}"
        )

    else:
        print(
            "JSON: chưa được tạo"
        )


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(
    start_stage: int = 1,
    end_stage: int | None = None,
    fresh: bool = False,
) -> None:

    check_scripts()

    total_stages = len(STAGES)

    if end_stage is None:
        end_stage = total_stages

    if not (
        1 <= start_stage <= total_stages
    ):
        raise ValueError(
            f"start-stage phải nằm trong "
            f"1 → {total_stages}"
        )

    if not (
        start_stage
        <= end_stage
        <= total_stages
    ):
        raise ValueError(
            f"end-stage phải nằm trong "
            f"{start_stage} → {total_stages}"
        )

    #
    # Chỉ stage 1 mới cần kiểm tra input_images.
    #
    if start_stage == 1:
        check_input_images()

    #
    # Fresh chỉ hợp lý khi chạy từ đầu.
    #
    if fresh:
        if start_stage != 1:
            raise ValueError(
                "--fresh chỉ được dùng khi "
                "--start-stage 1."
            )

        clean_generated_outputs()

    print()
    print("=" * 76)
    print("PASSPORT OCR END-TO-END PIPELINE")
    print("=" * 76)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Python       : {sys.executable}"
    )

    print(
        f"Stages       : "
        f"{start_stage} → {end_stage}"
    )

    print(
        f"Fresh run    : {fresh}"
    )
    print()
    print_device_summary()
    selected_stages = STAGES[
        start_stage - 1:
        end_stage
    ]

    pipeline_start = (
        time.time()
    )

    for stage_number, stage in enumerate(
        selected_stages,
        start=start_stage,
    ):
        run_stage(
            stage_number=stage_number,
            total_stages=total_stages,
            name=stage["name"],
            script=stage["script"],
        )

    total_elapsed = (
        time.time()
        - pipeline_start
    )

    print()
    print("=" * 76)
    print("PIPELINE HOÀN THÀNH")
    print("=" * 76)

    print(
        f"Tổng thời gian: "
        f"{total_elapsed:.1f}s"
    )

    if end_stage == total_stages:
        print_final_output()


# ============================================================
# LIST STAGES
# ============================================================

def print_stage_list() -> None:

    print(
        "PASSPORT OCR PIPELINE STAGES"
    )

    print(
        "=" * 76
    )

    for index, stage in enumerate(
        STAGES,
        start=1,
    ):
        print(
            f"{index:>2}. "
            f"{stage['name']:<34} "
            f"{stage['script']}"
        )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Passport OCR end-to-end pipeline."
        )
    )

    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        help=(
            "Stage bắt đầu. "
            "Mặc định: 1"
        ),
    )

    parser.add_argument(
        "--end-stage",
        type=int,
        default=None,
        help=(
            "Stage kết thúc. "
            "Mặc định: stage cuối."
        ),
    )

    parser.add_argument(
        "--list-stages",
        action="store_true",
        help=(
            "In danh sách pipeline stages "
            "và thoát."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Xóa output/checkpoint cũ "
            "trước khi chạy từ stage 1."
        ),
    )

    args = parser.parse_args()

    if args.list_stages:
        print_stage_list()
        return

    run_pipeline(
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()