from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from td3_parser import parse_td3


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "mrz_ocr"
    / "mrz_ocr_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mrz_parsed"
OUTPUT_CSV = OUTPUT_DIR / "mrz_parsed_results.csv"


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không thấy file OCR:\n{csv_path}"
        )

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def parse_one_row(
    row: dict[str, str],
) -> dict[str, Any]:
    filename = row.get("filename", "")

    line_1 = row.get("selected_line_1", "") or ""
    line_2 = row.get("selected_line_2", "") or ""

    line_1_length = len(line_1)
    line_2_length = len(line_2)

    if not line_1 or not line_2:
        return {
            "filename": filename,
            "ocr_selected_variant": row.get(
                "selected_variant"
            ),
            "ocr_mean_confidence": row.get(
                "selected_mean_confidence"
            ),
            "ocr_line_1": line_1,
            "ocr_line_2": line_2,
            "ocr_line_1_length": line_1_length,
            "ocr_line_2_length": line_2_length,
            "parse_status": "not_parsed",
            "parse_mode": None,
            "parse_error": "Thiếu một hoặc cả hai dòng MRZ.",
        }

    exact_length = (
        line_1_length == 44
        and line_2_length == 44
    )

    # Nếu đúng 44 + 44, parse strict.
    # Nếu lệch độ dài, vẫn parse mềm để ghi nhận dữ liệu,
    # nhưng phải đánh dấu rõ là padded_or_truncated.
    parsed = parse_td3(
        line_1=line_1,
        line_2=line_2,
        strict_length=exact_length,
    )

    parse_mode = (
        "strict_44_44"
        if exact_length
        else "padded_or_truncated"
    )

    parsed_dict = parsed.to_dict()

    return {
        "filename": filename,

        "ocr_selected_variant": row.get(
            "selected_variant"
        ),
        "ocr_mean_confidence": row.get(
            "selected_mean_confidence"
        ),

        "ocr_line_1": line_1,
        "ocr_line_2": line_2,
        "ocr_line_1_length": line_1_length,
        "ocr_line_2_length": line_2_length,

        "parse_mode": parse_mode,

        "document_type": parsed_dict[
            "document_type"
        ],
        "issuing_country": parsed_dict[
            "issuing_country"
        ],
        "surname": parsed_dict["surname"],
        "given_names": parsed_dict[
            "given_names"
        ],

        "passport_number": parsed_dict[
            "passport_number"
        ],
        "passport_number_check_digit": (
            parsed_dict[
                "passport_number_check_digit"
            ]
        ),

        "nationality": parsed_dict[
            "nationality"
        ],

        "birth_date_raw": parsed_dict[
            "birth_date_raw"
        ],
        "birth_date": parsed_dict[
            "birth_date"
        ],
        "birth_date_check_digit": (
            parsed_dict[
                "birth_date_check_digit"
            ]
        ),

        "sex": parsed_dict["sex"],

        "expiry_date_raw": parsed_dict[
            "expiry_date_raw"
        ],
        "expiry_date": parsed_dict[
            "expiry_date"
        ],
        "expiry_date_check_digit": (
            parsed_dict[
                "expiry_date_check_digit"
            ]
        ),

        "personal_number": parsed_dict[
            "personal_number"
        ],
        "personal_number_check_digit": (
            parsed_dict[
                "personal_number_check_digit"
            ]
        ),

        "final_check_digit": parsed_dict[
            "final_check_digit"
        ],

        "parse_status": parsed_dict[
            "parse_status"
        ],
        "parse_error": parsed_dict[
            "parse_error"
        ],
    }


def write_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
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

    with output_path.open(
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


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_rows = load_rows(INPUT_CSV)

    parsed_rows = [
        parse_one_row(row)
        for row in input_rows
    ]

    write_csv(
        parsed_rows,
        OUTPUT_CSV,
    )

    total = len(parsed_rows)

    strict_count = sum(
        1
        for row in parsed_rows
        if row.get("parse_mode") == "strict_44_44"
    )

    soft_count = sum(
        1
        for row in parsed_rows
        if row.get("parse_mode")
        == "padded_or_truncated"
    )

    not_parsed_count = sum(
        1
        for row in parsed_rows
        if row.get("parse_status") == "not_parsed"
    )

    parser_error_count = sum(
        1
        for row in parsed_rows
        if row.get("parse_status") == "error"
    )

    success_count = sum(
        1
        for row in parsed_rows
        if row.get("parse_status") == "success"
    )

    print("=" * 72)
    print("KẾT QUẢ PARSE MRZ TD3")
    print("=" * 72)

    print(f"Tổng record OCR             : {total}")
    print(f"Parse thành công            : {success_count}")
    print(f"Strict 44 + 44              : {strict_count}")
    print(f"Parse mềm                   : {soft_count}")
    print(f"Không đủ 2 dòng             : {not_parsed_count}")
    print(f"Parser error                : {parser_error_count}")

    print("\nOutput:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()