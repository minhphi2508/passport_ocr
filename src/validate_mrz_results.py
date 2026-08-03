from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from td3_validator import validate_td3_lines


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "mrz_parsed"
    / "mrz_parsed_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mrz_validated"
OUTPUT_CSV = OUTPUT_DIR / "mrz_validated_results.csv"


def load_rows(
    csv_path: Path,
) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không thấy file input:\n{csv_path}"
        )

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def write_rows(
    rows: list[dict[str, Any]],
    csv_path: Path,
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

    with csv_path.open(
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


def validate_one_row(
    row: dict[str, str],
) -> dict[str, Any]:
    line_1 = row.get("ocr_line_1", "") or ""
    line_2 = row.get("ocr_line_2", "") or ""

    validation = validate_td3_lines(
        line_1=line_1,
        line_2=line_2,
    )

    result = dict(row)

    for key, value in validation.to_dict().items():
        result[key] = value

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_rows = load_rows(INPUT_CSV)

    output_rows = [
        validate_one_row(row)
        for row in input_rows
    ]

    write_rows(
        output_rows,
        OUTPUT_CSV,
    )

    total = len(output_rows)

    validation_success = sum(
        1
        for row in output_rows
        if row.get("validation_status") == "success"
    )

    skipped_invalid_length = sum(
        1
        for row in output_rows
        if row.get("validation_status")
        == "skipped_invalid_length"
    )

    all_valid_true = sum(
        1
        for row in output_rows
        if str(row.get("all_main_checks_valid")) == "True"
    )

    all_valid_false = sum(
        1
        for row in output_rows
        if str(row.get("all_main_checks_valid")) == "False"
    )

    all_valid_unknown = sum(
        1
        for row in output_rows
        if row.get("all_main_checks_valid") is None
        or str(row.get("all_main_checks_valid")) in {"", "None"}
    )

    passport_failed = sum(
        1
        for row in output_rows
        if str(row.get("passport_number_check_valid")) == "False"
    )

    birth_failed = sum(
        1
        for row in output_rows
        if str(row.get("birth_date_check_valid")) == "False"
    )

    expiry_failed = sum(
        1
        for row in output_rows
        if str(row.get("expiry_date_check_valid")) == "False"
    )

    final_failed = sum(
        1
        for row in output_rows
        if str(row.get("final_check_valid")) == "False"
    )

    print("=" * 72)
    print("KẾT QUẢ CHECKSUM TD3")
    print("=" * 72)

    print(f"Tổng record                     : {total}")
    print(f"Validation chạy được            : {validation_success}")
    print(f"Skip vì không đúng 44 + 44      : {skipped_invalid_length}")

    print()
    print(f"All main checks = True          : {all_valid_true}")
    print(f"All main checks = False         : {all_valid_false}")
    print(f"All main checks = Unknown       : {all_valid_unknown}")

    print()
    print(f"Passport number check fail      : {passport_failed}")
    print(f"Birth date check fail           : {birth_failed}")
    print(f"Expiry date check fail          : {expiry_failed}")
    print(f"Final composite check fail      : {final_failed}")

    print("\nOutput:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()