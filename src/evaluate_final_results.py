from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINAL_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "final_results"
    / "passport_extraction_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "end_to_end_summary.csv"
)

DETAIL_CSV = (
    OUTPUT_DIR
    / "end_to_end_details.csv"
)


# ============================================================
# FINAL FIELDS
# ============================================================

FINAL_FIELDS = [
    "passport_number",
    "surname",
    "given_names",
    "nationality",
    "date_of_birth",
    "sex",
    "date_of_expiry",
    "date_of_issue",
]


# ============================================================
# HELPERS
# ============================================================

def load_rows(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy final results:\n{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def is_available(
    value: Any,
) -> bool:
    if value is None:
        return False

    text = str(value).strip()

    if not text:
        return False

    if text.lower() in {
        "none",
        "null",
        "nan",
    }:
        return False

    return True


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
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
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_rows(
        FINAL_CSV
    )

    total = len(rows)

    if total == 0:
        raise RuntimeError(
            "Final results đang rỗng."
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    complete = sum(
        1
        for row in rows
        if row.get(
            "final_status"
        ) == "complete"
    )

    partial = sum(
        1
        for row in rows
        if row.get(
            "final_status"
        ) == "partial"
    )

    failed = sum(
        1
        for row in rows
        if row.get(
            "final_status"
        ) == "failed"
    )

    # --------------------------------------------------------
    # MRZ
    # --------------------------------------------------------

    mrz_parsed = sum(
        1
        for row in rows
        if row.get(
            "mrz_parse_status"
        ) == "success"
    )

    mrz_validation_ran = sum(
        1
        for row in rows
        if row.get(
            "mrz_validation_status"
        ) == "success"
    )

    mrz_all_checks_valid = sum(
        1
        for row in rows
        if str(
            row.get(
                "all_main_checks_valid"
            )
        ).strip().lower()
        == "true"
    )

    # --------------------------------------------------------
    # DOI
    # --------------------------------------------------------

    doi_available = sum(
        1
        for row in rows
        if is_available(
            row.get(
                "date_of_issue"
            )
        )
    )

    doi_high_conf = sum(
        1
        for row in rows
        if row.get(
            "doi_status"
        ) == "high_confidence"
    )

    doi_medium_conf = sum(
        1
        for row in rows
        if row.get(
            "doi_status"
        ) == "medium_confidence"
    )

    doi_low_conf = sum(
        1
        for row in rows
        if row.get(
            "doi_status"
        ) == "low_confidence"
    )

    # --------------------------------------------------------
    # Field availability
    # --------------------------------------------------------

    field_counts = {}

    for field in FINAL_FIELDS:
        field_counts[field] = sum(
            1
            for row in rows
            if is_available(
                row.get(field)
            )
        )

    # --------------------------------------------------------
    # Detail rows
    # --------------------------------------------------------

    detail_rows = []

    for row in rows:
        missing_fields = [
            field
            for field in FINAL_FIELDS
            if not is_available(
                row.get(field)
            )
        ]

        detail_rows.append(
            {
                "filename": row.get(
                    "filename"
                ),
                "final_status": row.get(
                    "final_status"
                ),
                "mrz_parse_status": row.get(
                    "mrz_parse_status"
                ),
                "mrz_validation_status": row.get(
                    "mrz_validation_status"
                ),
                "all_main_checks_valid": row.get(
                    "all_main_checks_valid"
                ),
                "doi_status": row.get(
                    "doi_status"
                ),
                "missing_field_count": len(
                    missing_fields
                ),
                "missing_fields": " | ".join(
                    missing_fields
                ),
            }
        )

    write_csv(
        DETAIL_CSV,
        detail_rows,
    )

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary_rows = [
        {
            "metric": "total_images",
            "count": total,
            "rate": 1.0,
        },
        {
            "metric": "complete_records",
            "count": complete,
            "rate": complete / total,
        },
        {
            "metric": "partial_records",
            "count": partial,
            "rate": partial / total,
        },
        {
            "metric": "failed_records",
            "count": failed,
            "rate": failed / total,
        },
        {
            "metric": "mrz_parsed",
            "count": mrz_parsed,
            "rate": mrz_parsed / total,
        },
        {
            "metric": "mrz_validation_ran",
            "count": mrz_validation_ran,
            "rate": mrz_validation_ran / total,
        },
        {
            "metric": "mrz_all_checks_valid",
            "count": mrz_all_checks_valid,
            "rate": mrz_all_checks_valid / total,
        },
        {
            "metric": "date_of_issue_available",
            "count": doi_available,
            "rate": doi_available / total,
        },
    ]

    for field in FINAL_FIELDS:
        summary_rows.append(
            {
                "metric": (
                    f"field_available__{field}"
                ),
                "count": field_counts[
                    field
                ],
                "rate": (
                    field_counts[field]
                    / total
                ),
            }
        )

    write_csv(
        SUMMARY_CSV,
        summary_rows,
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("=" * 76)
    print("END-TO-END PASSPORT OCR EVALUATION")
    print("=" * 76)

    print(
        f"Total images                  : "
        f"{total}"
    )

    print()
    print(
        f"Complete records              : "
        f"{complete}/{total} "
        f"({complete / total:.1%})"
    )

    print(
        f"Partial records               : "
        f"{partial}/{total} "
        f"({partial / total:.1%})"
    )

    print(
        f"Failed records                : "
        f"{failed}/{total} "
        f"({failed / total:.1%})"
    )

    print()
    print(
        f"MRZ parsed                    : "
        f"{mrz_parsed}/{total} "
        f"({mrz_parsed / total:.1%})"
    )

    print(
        f"MRZ validation ran            : "
        f"{mrz_validation_ran}/{total} "
        f"({mrz_validation_ran / total:.1%})"
    )

    print(
        f"MRZ all main checks valid     : "
        f"{mrz_all_checks_valid}/{total} "
        f"({mrz_all_checks_valid / total:.1%})"
    )

    print()
    print(
        f"Date of Issue available       : "
        f"{doi_available}/{total} "
        f"({doi_available / total:.1%})"
    )

    print(
        f"DOI high confidence           : "
        f"{doi_high_conf}"
    )

    print(
        f"DOI medium confidence         : "
        f"{doi_medium_conf}"
    )

    print(
        f"DOI low confidence            : "
        f"{doi_low_conf}"
    )

    print()
    print("Field availability:")

    for field in FINAL_FIELDS:
        count = field_counts[
            field
        ]

        print(
            f"{field:<28}: "
            f"{count}/{total} "
            f"({count / total:.1%})"
        )

    print()
    print("Cases with missing fields:")

    missing_cases = [
        row
        for row in detail_rows
        if row[
            "missing_field_count"
        ] > 0
    ]

    if not missing_cases:
        print("None")

    else:
        for row in missing_cases:
            print(
                f"- {row['filename']} "
                f"-> {row['missing_fields']}"
            )

    print()
    print("=" * 76)
    print(
        "Lưu ý: đây là availability / pipeline success, "
        "KHÔNG phải field accuracy."
    )
    print("=" * 76)

    print()
    print("Summary CSV:")
    print(SUMMARY_CSV)

    print()
    print("Detail CSV:")
    print(DETAIL_CSV)


if __name__ == "__main__":
    main()