from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MRZ_VALIDATED_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "mrz_validated"
    / "mrz_validated_results.csv"
)

DOI_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "date_of_issue_hybrid_v3"
    / "date_of_issue_hybrid_v3_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_results"

OUTPUT_CSV = OUTPUT_DIR / "passport_extraction_results.csv"
OUTPUT_JSON = OUTPUT_DIR / "passport_extraction_results.json"


# ============================================================
# HELPERS
# ============================================================

def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy file:\n{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


def empty_to_none(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()

        if stripped.lower() in {
            "",
            "none",
            "nan",
            "null",
        }:
            return None

        return stripped

    return value


def parse_optional_bool(
    value: Any,
) -> bool | None:
    normalized = empty_to_none(value)

    if normalized is None:
        return None

    text = str(normalized).strip().lower()

    if text == "true":
        return True

    if text == "false":
        return False

    return None


def parse_optional_float(
    value: Any,
) -> float | None:
    normalized = empty_to_none(value)

    if normalized is None:
        return None

    try:
        return float(normalized)

    except (TypeError, ValueError):
        return None


def parse_candidate_dates(
    value: Any,
) -> list[str]:
    normalized = empty_to_none(value)

    if normalized is None:
        return []

    return [
        item.strip()
        for item in str(normalized).split("|")
        if item.strip()
    ]


# ============================================================
# BUILD ONE FINAL RECORD
# ============================================================

def build_final_record(
    filename: str,
    mrz_row: dict[str, str] | None,
    doi_row: dict[str, str] | None,
) -> dict[str, Any]:
    mrz_row = mrz_row or {}
    doi_row = doi_row or {}

    parse_status = empty_to_none(
        mrz_row.get("parse_status")
    )

    validation_status = empty_to_none(
        mrz_row.get("validation_status")
    )

    doi_status = empty_to_none(
        doi_row.get("status")
    )

    date_of_issue = empty_to_none(
        doi_row.get("date_of_issue")
    )

    mrz_available = bool(mrz_row)
    doi_available = date_of_issue is not None

    final_status_reasons: list[str] = []

    if not mrz_available:
        final_status_reasons.append(
            "mrz_result_missing"
        )

    elif parse_status != "success":
        final_status_reasons.append(
            "mrz_parse_not_success"
        )

    if not doi_available:
        final_status_reasons.append(
            "date_of_issue_missing"
        )

    if not final_status_reasons:
        final_status = "complete"

    elif mrz_available:
        final_status = "partial"

    else:
        final_status = "failed"

    record = {
        # ----------------------------------------------------
        # ID
        # ----------------------------------------------------
        "filename": filename,
        "final_status": final_status,
        "final_status_reasons": final_status_reasons,

        # ----------------------------------------------------
        # FINAL FIELDS
        # ----------------------------------------------------
        "passport_number": empty_to_none(
            mrz_row.get("passport_number")
        ),

        "surname": empty_to_none(
            mrz_row.get("surname")
        ),

        "given_names": empty_to_none(
            mrz_row.get("given_names")
        ),

        "nationality": empty_to_none(
            mrz_row.get("nationality")
        ),

        "date_of_birth": empty_to_none(
            mrz_row.get("birth_date")
        ),

        "sex": empty_to_none(
            mrz_row.get("sex")
        ),

        "date_of_expiry": empty_to_none(
            mrz_row.get("expiry_date")
        ),

        "date_of_issue": date_of_issue,

        # ----------------------------------------------------
        # ADDITIONAL MRZ FIELDS
        # ----------------------------------------------------
        "document_type": empty_to_none(
            mrz_row.get("document_type")
        ),

        "issuing_country": empty_to_none(
            mrz_row.get("issuing_country")
        ),

        "personal_number": empty_to_none(
            mrz_row.get("personal_number")
        ),

        # ----------------------------------------------------
        # OCR METADATA
        # ----------------------------------------------------
        "mrz_ocr_variant": empty_to_none(
            mrz_row.get("ocr_selected_variant")
        ),

        "mrz_ocr_confidence": (
            parse_optional_float(
                mrz_row.get(
                    "ocr_mean_confidence"
                )
            )
        ),

        "mrz_line_1": empty_to_none(
            mrz_row.get("ocr_line_1")
        ),

        "mrz_line_2": empty_to_none(
            mrz_row.get("ocr_line_2")
        ),

        "mrz_parse_mode": empty_to_none(
            mrz_row.get("parse_mode")
        ),

        "mrz_parse_status": parse_status,

        # ----------------------------------------------------
        # CHECKSUM METADATA
        # ----------------------------------------------------
        "mrz_validation": {
            "validation_status": (
                validation_status
            ),

            "passport_number_check_valid": (
                parse_optional_bool(
                    mrz_row.get(
                        "passport_number_check_valid"
                    )
                )
            ),

            "birth_date_check_valid": (
                parse_optional_bool(
                    mrz_row.get(
                        "birth_date_check_valid"
                    )
                )
            ),

            "expiry_date_check_valid": (
                parse_optional_bool(
                    mrz_row.get(
                        "expiry_date_check_valid"
                    )
                )
            ),

            "personal_number_check_valid": (
                parse_optional_bool(
                    mrz_row.get(
                        "personal_number_check_valid"
                    )
                )
            ),

            "final_check_valid": (
                parse_optional_bool(
                    mrz_row.get(
                        "final_check_valid"
                    )
                )
            ),

            "all_main_checks_valid": (
                parse_optional_bool(
                    mrz_row.get(
                        "all_main_checks_valid"
                    )
                )
            ),
        },

        # ----------------------------------------------------
        # DOI METADATA
        # ----------------------------------------------------
        "date_of_issue_metadata": {
            "status": doi_status,

            "method": empty_to_none(
                doi_row.get("method")
            ),

            "score": parse_optional_float(
                doi_row.get("score")
            ),

            "score_margin": (
                parse_optional_float(
                    doi_row.get(
                        "score_margin"
                    )
                )
            ),

            "label_text": empty_to_none(
                doi_row.get("label_text")
            ),

            "candidate_text": empty_to_none(
                doi_row.get(
                    "candidate_text"
                )
            ),

            "all_candidate_dates": (
                parse_candidate_dates(
                    doi_row.get(
                        "all_candidate_dates"
                    )
                )
            ),
        },
    }

    return record


# ============================================================
# FLATTEN FOR CSV
# ============================================================

def flatten_record_for_csv(
    record: dict[str, Any],
) -> dict[str, Any]:
    validation = record[
        "mrz_validation"
    ]

    doi_metadata = record[
        "date_of_issue_metadata"
    ]

    return {
        "filename": record["filename"],
        "final_status": record[
            "final_status"
        ],

        "final_status_reasons": " | ".join(
            record[
                "final_status_reasons"
            ]
        ),

        "passport_number": record[
            "passport_number"
        ],

        "surname": record["surname"],

        "given_names": record[
            "given_names"
        ],

        "nationality": record[
            "nationality"
        ],

        "date_of_birth": record[
            "date_of_birth"
        ],

        "sex": record["sex"],

        "date_of_expiry": record[
            "date_of_expiry"
        ],

        "date_of_issue": record[
            "date_of_issue"
        ],

        "document_type": record[
            "document_type"
        ],

        "issuing_country": record[
            "issuing_country"
        ],

        "personal_number": record[
            "personal_number"
        ],

        "mrz_ocr_variant": record[
            "mrz_ocr_variant"
        ],

        "mrz_ocr_confidence": record[
            "mrz_ocr_confidence"
        ],

        "mrz_parse_mode": record[
            "mrz_parse_mode"
        ],

        "mrz_parse_status": record[
            "mrz_parse_status"
        ],

        "mrz_validation_status": validation[
            "validation_status"
        ],

        "passport_number_check_valid": (
            validation[
                "passport_number_check_valid"
            ]
        ),

        "birth_date_check_valid": (
            validation[
                "birth_date_check_valid"
            ]
        ),

        "expiry_date_check_valid": (
            validation[
                "expiry_date_check_valid"
            ]
        ),

        "personal_number_check_valid": (
            validation[
                "personal_number_check_valid"
            ]
        ),

        "final_check_valid": validation[
            "final_check_valid"
        ],

        "all_main_checks_valid": (
            validation[
                "all_main_checks_valid"
            ]
        ),

        "doi_status": doi_metadata[
            "status"
        ],

        "doi_method": doi_metadata[
            "method"
        ],

        "doi_score": doi_metadata[
            "score"
        ],

        "doi_score_margin": doi_metadata[
            "score_margin"
        ],

        "doi_label_text": doi_metadata[
            "label_text"
        ],

        "doi_candidate_text": doi_metadata[
            "candidate_text"
        ],

        "doi_all_candidate_dates": " | ".join(
            doi_metadata[
                "all_candidate_dates"
            ]
        ),

        "mrz_line_1": record[
            "mrz_line_1"
        ],

        "mrz_line_2": record[
            "mrz_line_2"
        ],
    }


# ============================================================
# WRITE OUTPUTS
# ============================================================

def write_json(
    records: list[dict[str, Any]],
) -> None:
    with OUTPUT_JSON.open(
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
    rows = [
        flatten_record_for_csv(record)
        for record in records
    ]

    if not rows:
        return

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mrz_rows = load_csv(
        MRZ_VALIDATED_CSV
    )

    doi_rows = load_csv(
        DOI_CSV
    )

    mrz_by_filename = {
        row["filename"]: row
        for row in mrz_rows
        if row.get("filename")
    }

    doi_by_filename = {
        row["filename"]: row
        for row in doi_rows
        if row.get("filename")
    }

    all_filenames = sorted(
        set(mrz_by_filename)
        | set(doi_by_filename)
    )

    records = [
        build_final_record(
            filename=filename,
            mrz_row=mrz_by_filename.get(
                filename
            ),
            doi_row=doi_by_filename.get(
                filename
            ),
        )
        for filename in all_filenames
    ]

    write_json(records)
    write_csv(records)

    complete_count = sum(
        1
        for record in records
        if record["final_status"]
        == "complete"
    )

    partial_count = sum(
        1
        for record in records
        if record["final_status"]
        == "partial"
    )

    failed_count = sum(
        1
        for record in records
        if record["final_status"]
        == "failed"
    )

    print("=" * 76)
    print("KẾT QUẢ TỔNG HỢP CUỐI")
    print("=" * 76)

    print(
        f"Tổng ảnh có kết quả        : "
        f"{len(records)}"
    )

    print(
        f"Complete                   : "
        f"{complete_count}"
    )

    print(
        f"Partial                    : "
        f"{partial_count}"
    )

    print(
        f"Failed                     : "
        f"{failed_count}"
    )

    print("\nCác trường output chính:")

    print(
        "passport_number, surname, given_names, "
        "nationality, date_of_birth, sex, "
        "date_of_expiry, date_of_issue"
    )

    print("\nCSV:")
    print(OUTPUT_CSV)

    print("\nJSON:")
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()