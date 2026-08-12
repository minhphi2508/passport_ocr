from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# MRZ
# ------------------------------------------------------------

MRZ_VALIDATED_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "mrz_validated"
    / "mrz_validated_results.csv"
)


# ------------------------------------------------------------
# VIZ structured fields
# ------------------------------------------------------------

VIZ_FIELDS_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "viz_fields"
    / "viz_fields_results.csv"
)


# ------------------------------------------------------------
# Date of Issue
# ------------------------------------------------------------

DOI_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "date_of_issue_hybrid_v3"
    / "date_of_issue_hybrid_v3_results.csv"
)


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_results"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "passport_extraction_results.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "passport_extraction_results.json"
)


# ============================================================
# FINAL FIELDS
# ============================================================

CORE_FIELDS = (
    "passport_number",
    "surname",
    "given_names",
    "nationality",
    "date_of_birth",
    "sex",
    "date_of_expiry",
    "date_of_issue",
)


# ============================================================
# CSV HELPERS
# ============================================================

def load_csv(
    path: Path,
    required: bool = True,
) -> list[dict[str, str]]:

    if not path.exists():

        if required:

            raise FileNotFoundError(
                f"Không thấy file:\n"
                f"{path}"
            )

        return []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


def empty_to_none(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        str,
    ):

        stripped = (
            value.strip()
        )

        if (
            stripped.lower()
            in {
                "",
                "none",
                "nan",
                "null",
            }
        ):
            return None

        return stripped

    return value


def parse_optional_bool(
    value: Any,
) -> bool | None:

    normalized = empty_to_none(
        value
    )

    if normalized is None:
        return None

    text = str(
        normalized
    ).strip().lower()

    if text == "true":
        return True

    if text == "false":
        return False

    return None


def parse_optional_float(
    value: Any,
) -> float | None:

    normalized = empty_to_none(
        value
    )

    if normalized is None:
        return None

    try:

        return float(
            normalized
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


def parse_optional_int(
    value: Any,
) -> int | None:

    normalized = empty_to_none(
        value
    )

    if normalized is None:
        return None

    try:

        return int(
            float(
                normalized
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


def parse_candidate_dates(
    value: Any,
) -> list[str]:

    normalized = empty_to_none(
        value
    )

    if normalized is None:
        return []

    return [
        item.strip()
        for item in str(
            normalized
        ).split("|")
        if item.strip()
    ]


# ============================================================
# FIELD SELECTION
# ============================================================

def choose_field(
    mrz_value: Any,
    viz_value: Any,
    mrz_parse_success: bool,
) -> tuple[
    Any,
    str | None,
]:
    """
    Source priority:

    1. MRZ khi MRZ parse success.
    2. VIZ.
    3. MRZ fallback nếu parse chưa success nhưng
       field vẫn tồn tại.

    Rule (3) giữ behavior hiện tại nhưng đánh source
    rõ ràng để review sau.
    """

    mrz_value = empty_to_none(
        mrz_value
    )

    viz_value = empty_to_none(
        viz_value
    )

    if (
        mrz_parse_success
        and mrz_value is not None
    ):

        return (
            mrz_value,
            "mrz",
        )

    if (
        viz_value
        is not None
    ):

        return (
            viz_value,
            "viz",
        )

    if (
        mrz_value
        is not None
    ):

        return (
            mrz_value,
            "mrz_unvalidated",
        )

    return (
        None,
        None,
    )


# ============================================================
# BUILD ONE RECORD
# ============================================================

def build_final_record(
    filename: str,
    mrz_row:
        dict[str, str]
        | None,
    viz_row:
        dict[str, str]
        | None,
    doi_row:
        dict[str, str]
        | None,
) -> dict[str, Any]:

    mrz_row = (
        mrz_row
        or {}
    )

    viz_row = (
        viz_row
        or {}
    )

    doi_row = (
        doi_row
        or {}
    )

    # ========================================================
    # AVAILABILITY
    # ========================================================

    mrz_available = bool(
        mrz_row
    )

    viz_available = bool(
        viz_row
    )

    parse_status = (
        empty_to_none(
            mrz_row.get(
                "parse_status"
            )
        )
    )

    validation_status = (
        empty_to_none(
            mrz_row.get(
                "validation_status"
            )
        )
    )

    mrz_parse_success = (
        parse_status
        == "success"
    )

    all_main_checks_valid = (
        parse_optional_bool(
            mrz_row.get(
                "all_main_checks_valid"
            )
        )
    )

    # ========================================================
    # FUSE NORMAL FIELDS
    # ========================================================

    field_sources: dict[
        str,
        str | None,
    ] = {}

    (
        passport_number,
        field_sources[
            "passport_number"
        ],
    ) = choose_field(
        mrz_value=(
            mrz_row.get(
                "passport_number"
            )
        ),
        viz_value=(
            viz_row.get(
                "passport_number"
            )
        ),
        mrz_parse_success=(
            mrz_parse_success
        ),
    )

    (
        surname,
        field_sources[
            "surname"
        ],
    ) = choose_field(
        mrz_value=(
            mrz_row.get(
                "surname"
            )
        ),
        viz_value=(
            viz_row.get(
                "surname"
            )
        ),
        mrz_parse_success=(
            mrz_parse_success
        ),
    )

    (
        given_names,
        field_sources[
            "given_names"
        ],
    ) = choose_field(
        mrz_value=(
            mrz_row.get(
                "given_names"
            )
        ),
        viz_value=(
            viz_row.get(
                "given_names"
            )
        ),
        mrz_parse_success=(
            mrz_parse_success
        ),
    )

    (
        nationality,
        field_sources[
            "nationality"
        ],
    ) = choose_field(
        mrz_value=(
            mrz_row.get(
                "nationality"
            )
        ),
        viz_value=(
            viz_row.get(
                "nationality"
            )
        ),
        mrz_parse_success=(
            mrz_parse_success
        ),
    )

    (
        date_of_birth,
        field_sources[
            "date_of_birth"
        ],
    ) = choose_field(
        mrz_value=(
            mrz_row.get(
                "birth_date"
            )
        ),
        viz_value=(
            viz_row.get(
                "date_of_birth"
            )
        ),
        mrz_parse_success=(
            mrz_parse_success
        ),
    )

    (
        sex,
        field_sources[
            "sex"
        ],
    ) = choose_field(
        mrz_value=(
            mrz_row.get(
                "sex"
            )
        ),
        viz_value=(
            viz_row.get(
                "sex"
            )
        ),
        mrz_parse_success=(
            mrz_parse_success
        ),
    )

    (
        date_of_expiry,
        field_sources[
            "date_of_expiry"
        ],
    ) = choose_field(
        mrz_value=(
            mrz_row.get(
                "expiry_date"
            )
        ),
        viz_value=(
            viz_row.get(
                "date_of_expiry"
            )
        ),
        mrz_parse_success=(
            mrz_parse_success
        ),
    )

    # ========================================================
    # DOI
    # ========================================================

    date_of_issue = (
        empty_to_none(
            doi_row.get(
                "date_of_issue"
            )
        )
    )

    field_sources[
        "date_of_issue"
    ] = (
        "viz_doi"
        if date_of_issue
        is not None
        else None
    )

    # ========================================================
    # EXTRACTION MODE
    # ========================================================

    if (
        mrz_available
        and viz_available
    ):

        extraction_mode = (
            "mrz_plus_viz"
        )

    elif (
        mrz_available
    ):

        extraction_mode = (
            "mrz_only"
        )

    elif (
        viz_available
    ):

        extraction_mode = (
            "viz_only"
        )

    else:

        extraction_mode = (
            "none"
        )

    # ========================================================
    # FINAL FIELD VALUES
    # ========================================================

    final_fields = {
        "passport_number":
            passport_number,

        "surname":
            surname,

        "given_names":
            given_names,

        "nationality":
            nationality,

        "date_of_birth":
            date_of_birth,

        "sex":
            sex,

        "date_of_expiry":
            date_of_expiry,

        "date_of_issue":
            date_of_issue,
    }

    missing_fields = [
        field_name
        for field_name
        in CORE_FIELDS
        if (
            final_fields[
                field_name
            ]
            is None
        )
    ]

    extracted_field_count = (
        len(
            CORE_FIELDS
        )
        - len(
            missing_fields
        )
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    if (
        extracted_field_count
        == len(
            CORE_FIELDS
        )
    ):

        final_status = (
            "complete"
        )

    elif (
        extracted_field_count
        > 0
    ):

        final_status = (
            "partial"
        )

    else:

        final_status = (
            "failed"
        )

    # ========================================================
    # STATUS REASONS
    # ========================================================

    final_status_reasons = []

    if not mrz_available:

        final_status_reasons.append(
            "mrz_unavailable"
        )

    elif not mrz_parse_success:

        final_status_reasons.append(
            "mrz_parse_not_success"
        )

    if (
        mrz_available
        and all_main_checks_valid
        is False
    ):

        final_status_reasons.append(
            "mrz_checksum_failed"
        )

    if missing_fields:

        final_status_reasons.append(
            "missing_fields:"
            + ",".join(
                missing_fields
            )
        )

    # ========================================================
    # REVIEW FLAG
    # ========================================================

    review_required = False

    review_reasons = []

    if (
        extraction_mode
        == "viz_only"
    ):

        review_required = True

        review_reasons.append(
            "viz_only_extraction"
        )

    if (
        parse_status
        not in {
            None,
            "success",
        }
    ):

        review_required = True

        review_reasons.append(
            "mrz_parse_not_success"
        )

    if (
        all_main_checks_valid
        is False
    ):

        review_required = True

        review_reasons.append(
            "mrz_checksum_failed"
        )

    if (
        final_status
        != "complete"
    ):

        review_required = True

        review_reasons.append(
            "incomplete_fields"
        )

    # ========================================================
    # RECORD
    # ========================================================

    return {
        # ----------------------------------------------------
        # ID / STATUS
        # ----------------------------------------------------

        "filename":
            filename,

        "final_status":
            final_status,

        "extraction_mode":
            extraction_mode,

        "extracted_field_count":
            extracted_field_count,

        "missing_fields":
            missing_fields,

        "final_status_reasons":
            final_status_reasons,

        "review_required":
            review_required,

        "review_reasons":
            review_reasons,

        # ----------------------------------------------------
        # FINAL FIELDS
        # ----------------------------------------------------

        **final_fields,

        # ----------------------------------------------------
        # FIELD SOURCE
        # ----------------------------------------------------

        "field_sources":
            field_sources,

        # ----------------------------------------------------
        # ADDITIONAL MRZ
        # ----------------------------------------------------

        "document_type":
            empty_to_none(
                mrz_row.get(
                    "document_type"
                )
            ),

        "issuing_country":
            empty_to_none(
                mrz_row.get(
                    "issuing_country"
                )
            ),

        "personal_number":
            empty_to_none(
                mrz_row.get(
                    "personal_number"
                )
            ),

        # ----------------------------------------------------
        # MRZ METADATA
        # ----------------------------------------------------

        "mrz_available":
            mrz_available,

        "mrz_parse_status":
            parse_status,

        "mrz_parse_mode":
            empty_to_none(
                mrz_row.get(
                    "parse_mode"
                )
            ),

        "mrz_ocr_variant":
            empty_to_none(
                mrz_row.get(
                    "ocr_selected_variant"
                )
            ),

        "mrz_ocr_confidence":
            parse_optional_float(
                mrz_row.get(
                    "ocr_mean_confidence"
                )
            ),

        "mrz_line_1":
            empty_to_none(
                mrz_row.get(
                    "ocr_line_1"
                )
            ),

        "mrz_line_2":
            empty_to_none(
                mrz_row.get(
                    "ocr_line_2"
                )
            ),

        # ----------------------------------------------------
        # MRZ VALIDATION
        # ----------------------------------------------------

        "mrz_validation": {
            "validation_status":
                validation_status,

            "passport_number_check_valid":
                parse_optional_bool(
                    mrz_row.get(
                        "passport_number_check_valid"
                    )
                ),

            "birth_date_check_valid":
                parse_optional_bool(
                    mrz_row.get(
                        "birth_date_check_valid"
                    )
                ),

            "expiry_date_check_valid":
                parse_optional_bool(
                    mrz_row.get(
                        "expiry_date_check_valid"
                    )
                ),

            "personal_number_check_valid":
                parse_optional_bool(
                    mrz_row.get(
                        "personal_number_check_valid"
                    )
                ),

            "final_check_valid":
                parse_optional_bool(
                    mrz_row.get(
                        "final_check_valid"
                    )
                ),

            "all_main_checks_valid":
                all_main_checks_valid,
        },

        # ----------------------------------------------------
        # VIZ METADATA
        # ----------------------------------------------------

        "viz_available":
            viz_available,

        "viz_extraction_status":
            empty_to_none(
                viz_row.get(
                    "viz_extraction_status"
                )
            ),

        "viz_extracted_field_count":
            parse_optional_int(
                viz_row.get(
                    "extracted_field_count"
                )
            ),

        "viz_selected_variant":
            empty_to_none(
                viz_row.get(
                    "viz_selected_variant"
                )
            ),

        # ----------------------------------------------------
        # DOI METADATA
        # ----------------------------------------------------

        "date_of_issue_metadata": {
            "status":
                empty_to_none(
                    doi_row.get(
                        "status"
                    )
                ),

            "method":
                empty_to_none(
                    doi_row.get(
                        "method"
                    )
                ),

            "score":
                parse_optional_float(
                    doi_row.get(
                        "score"
                    )
                ),

            "score_margin":
                parse_optional_float(
                    doi_row.get(
                        "score_margin"
                    )
                ),

            "label_text":
                empty_to_none(
                    doi_row.get(
                        "label_text"
                    )
                ),

            "candidate_text":
                empty_to_none(
                    doi_row.get(
                        "candidate_text"
                    )
                ),

            "all_candidate_dates":
                parse_candidate_dates(
                    doi_row.get(
                        "all_candidate_dates"
                    )
                ),
        },
    }


# ============================================================
# CSV FLATTEN
# ============================================================

def flatten_record_for_csv(
    record: dict[str, Any],
) -> dict[str, Any]:

    validation = (
        record[
            "mrz_validation"
        ]
    )

    doi_metadata = (
        record[
            "date_of_issue_metadata"
        ]
    )

    field_sources = (
        record[
            "field_sources"
        ]
    )

    return {
        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        "filename":
            record[
                "filename"
            ],

        "final_status":
            record[
                "final_status"
            ],

        "extraction_mode":
            record[
                "extraction_mode"
            ],

        "extracted_field_count":
            record[
                "extracted_field_count"
            ],

        "missing_fields":
            " | ".join(
                record[
                    "missing_fields"
                ]
            ),

        "final_status_reasons":
            " | ".join(
                record[
                    "final_status_reasons"
                ]
            ),

        "review_required":
            record[
                "review_required"
            ],

        "review_reasons":
            " | ".join(
                record[
                    "review_reasons"
                ]
            ),

        # ----------------------------------------------------
        # FIELDS
        # ----------------------------------------------------

        "passport_number":
            record[
                "passport_number"
            ],

        "surname":
            record[
                "surname"
            ],

        "given_names":
            record[
                "given_names"
            ],

        "nationality":
            record[
                "nationality"
            ],

        "date_of_birth":
            record[
                "date_of_birth"
            ],

        "sex":
            record[
                "sex"
            ],

        "date_of_expiry":
            record[
                "date_of_expiry"
            ],

        "date_of_issue":
            record[
                "date_of_issue"
            ],

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        "passport_number_source":
            field_sources.get(
                "passport_number"
            ),

        "surname_source":
            field_sources.get(
                "surname"
            ),

        "given_names_source":
            field_sources.get(
                "given_names"
            ),

        "nationality_source":
            field_sources.get(
                "nationality"
            ),

        "date_of_birth_source":
            field_sources.get(
                "date_of_birth"
            ),

        "sex_source":
            field_sources.get(
                "sex"
            ),

        "date_of_expiry_source":
            field_sources.get(
                "date_of_expiry"
            ),

        "date_of_issue_source":
            field_sources.get(
                "date_of_issue"
            ),

        # ----------------------------------------------------
        # ADDITIONAL MRZ
        # ----------------------------------------------------

        "document_type":
            record[
                "document_type"
            ],

        "issuing_country":
            record[
                "issuing_country"
            ],

        "personal_number":
            record[
                "personal_number"
            ],

        # ----------------------------------------------------
        # MRZ
        # ----------------------------------------------------

        "mrz_available":
            record[
                "mrz_available"
            ],

        "mrz_parse_status":
            record[
                "mrz_parse_status"
            ],

        "mrz_parse_mode":
            record[
                "mrz_parse_mode"
            ],

        "mrz_ocr_variant":
            record[
                "mrz_ocr_variant"
            ],

        "mrz_ocr_confidence":
            record[
                "mrz_ocr_confidence"
            ],

        "mrz_validation_status":
            validation[
                "validation_status"
            ],

        "passport_number_check_valid":
            validation[
                "passport_number_check_valid"
            ],

        "birth_date_check_valid":
            validation[
                "birth_date_check_valid"
            ],

        "expiry_date_check_valid":
            validation[
                "expiry_date_check_valid"
            ],

        "personal_number_check_valid":
            validation[
                "personal_number_check_valid"
            ],

        "final_check_valid":
            validation[
                "final_check_valid"
            ],

        "all_main_checks_valid":
            validation[
                "all_main_checks_valid"
            ],

        # ----------------------------------------------------
        # VIZ
        # ----------------------------------------------------

        "viz_available":
            record[
                "viz_available"
            ],

        "viz_extraction_status":
            record[
                "viz_extraction_status"
            ],

        "viz_extracted_field_count":
            record[
                "viz_extracted_field_count"
            ],

        "viz_selected_variant":
            record[
                "viz_selected_variant"
            ],

        # ----------------------------------------------------
        # DOI
        # ----------------------------------------------------

        "doi_status":
            doi_metadata[
                "status"
            ],

        "doi_method":
            doi_metadata[
                "method"
            ],

        "doi_score":
            doi_metadata[
                "score"
            ],

        "doi_score_margin":
            doi_metadata[
                "score_margin"
            ],

        "doi_label_text":
            doi_metadata[
                "label_text"
            ],

        "doi_candidate_text":
            doi_metadata[
                "candidate_text"
            ],

        "doi_all_candidate_dates":
            " | ".join(
                doi_metadata[
                    "all_candidate_dates"
                ]
            ),

        # ----------------------------------------------------
        # RAW MRZ
        # ----------------------------------------------------

        "mrz_line_1":
            record[
                "mrz_line_1"
            ],

        "mrz_line_2":
            record[
                "mrz_line_2"
            ],
    }


# ============================================================
# WRITE
# ============================================================

def write_json(
    records: list[
        dict[
            str,
            Any,
        ]
    ],
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
    records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:

    rows = [
        flatten_record_for_csv(
            record
        )
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

        writer.writerows(
            rows
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # MRZ có thể thiếu đối với VIZ-only passport.
    mrz_rows = load_csv(
        MRZ_VALIDATED_CSV,
        required=False,
    )

    # VIZ fields là stage mới.
    viz_rows = load_csv(
        VIZ_FIELDS_CSV,
        required=True,
    )

    doi_rows = load_csv(
        DOI_CSV,
        required=True,
    )

    mrz_by_filename = {
        row[
            "filename"
        ]:
            row
        for row in mrz_rows
        if row.get(
            "filename"
        )
    }

    viz_by_filename = {
        row[
            "filename"
        ]:
            row
        for row in viz_rows
        if row.get(
            "filename"
        )
    }

    doi_by_filename = {
        row[
            "filename"
        ]:
            row
        for row in doi_rows
        if row.get(
            "filename"
        )
    }

    # ========================================================
    # UNION:
    #
    # Đây là thay đổi quan trọng.
    #
    # Không còn lấy universe chỉ từ MRZ + DOI.
    # Passport chỉ có VIZ vẫn phải xuất hiện final.
    # ========================================================

    all_filenames = sorted(
        set(
            mrz_by_filename
        )
        | set(
            viz_by_filename
        )
        | set(
            doi_by_filename
        )
    )

    records = [
        build_final_record(
            filename=filename,

            mrz_row=(
                mrz_by_filename.get(
                    filename
                )
            ),

            viz_row=(
                viz_by_filename.get(
                    filename
                )
            ),

            doi_row=(
                doi_by_filename.get(
                    filename
                )
            ),
        )
        for filename
        in all_filenames
    ]

    write_json(
        records
    )

    write_csv(
        records
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    complete_count = sum(
        record[
            "final_status"
        ]
        == "complete"
        for record in records
    )

    partial_count = sum(
        record[
            "final_status"
        ]
        == "partial"
        for record in records
    )

    failed_count = sum(
        record[
            "final_status"
        ]
        == "failed"
        for record in records
    )

    mode_counts: dict[
        str,
        int,
    ] = {}

    for record in records:

        mode = (
            record[
                "extraction_mode"
            ]
        )

        mode_counts[
            mode
        ] = (
            mode_counts.get(
                mode,
                0,
            )
            + 1
        )

    review_count = sum(
        bool(
            record[
                "review_required"
            ]
        )
        for record in records
    )

    print(
        "=" * 76
    )

    print(
        "FINAL PASSPORT EXTRACTION"
    )

    print(
        "=" * 76
    )

    print(
        f"Tổng ảnh có kết quả : "
        f"{len(records)}"
    )

    print()

    print(
        "FINAL STATUS"
    )

    print(
        "-" * 76
    )

    print(
        f"Complete            : "
        f"{complete_count}"
    )

    print(
        f"Partial             : "
        f"{partial_count}"
    )

    print(
        f"Failed              : "
        f"{failed_count}"
    )

    print()

    print(
        "EXTRACTION MODE"
    )

    print(
        "-" * 76
    )

    for (
        mode,
        count,
    ) in sorted(
        mode_counts.items()
    ):

        print(
            f"{mode:<24}: "
            f"{count}"
        )

    print()

    print(
        f"Review required     : "
        f"{review_count}"
    )

    print()

    print(
        "CSV:"
    )

    print(
        OUTPUT_CSV
    )

    print()

    print(
        "JSON:"
    )

    print(
        OUTPUT_JSON
    )


if __name__ == "__main__":
    main()