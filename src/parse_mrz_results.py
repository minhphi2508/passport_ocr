from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from td3_parser import (
    correct_alpha_field_ocr,
    parse_td3,
)
from td3_validator import validate_td3_lines


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "mrz_ocr"
    / "mrz_ocr_results.csv"
)

INPUT_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "mrz_ocr"
    / "mrz_ocr_results.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mrz_parsed"
OUTPUT_CSV = OUTPUT_DIR / "mrz_parsed_results.csv"


# OCR confusions used only for checksum-guided repair.
# No correction is accepted unless structural/checksum evidence supports it.
PASSPORT_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "0": ("O",),
    "O": ("0",),
    "1": ("I",),
    "I": ("1",),
    "2": ("Z",),
    "Z": ("2",),
    "5": ("S",),
    "S": ("5",),
    "6": ("G",),
    "G": ("6",),
    "8": ("B",),
    "B": ("8",),
}


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


def load_ocr_records(
    json_path: Path,
) -> list[dict[str, Any]]:
    if not json_path.exists():
        return []

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        data = json.load(json_file)

    return data if isinstance(data, list) else []


def clean_line(value: Any) -> str:
    return (
        str(value or "")
        .upper()
        .strip()
        .replace(" ", "")
    )


def alpha_code(value: str) -> str:
    return correct_alpha_field_ocr(
        clean_line(value)
    ).replace("<", "")


def passport_value(line_2: str) -> str:
    return line_2[0:9].rstrip("<")


def passport_pattern(value: str) -> str:
    pattern = []

    for char in value:
        if char.isdigit():
            pattern.append("9")
        elif "A" <= char <= "Z":
            pattern.append("A")
        else:
            pattern.append("?")

    return "".join(pattern)


def issuer_from_line_1(line_1: str) -> str:
    if len(line_1) < 5:
        return ""

    return alpha_code(line_1[2:5])


def nationality_from_line_2(line_2: str) -> str:
    if len(line_2) < 13:
        return ""

    return alpha_code(line_2[10:13])


def repair_td3_line_1_prefix(
    line_1: str,
) -> tuple[str, str | None]:
    """
    Conservative TD3 framing repair.

    Passport TD3 line 1 must start with the document code P.
    If OCR prepended 1-2 garbage characters but an explicit 'P<'
    occurs immediately afterwards, trim only that garbage and pad
    the right side back to 44 characters.

    Example:
        EP<INDINAMDAR... -> P<INDINAMDAR...
    """
    line = clean_line(line_1)

    if len(line) == 44 and line.startswith("P"):
        return line, None

    for offset in (1, 2):
        if line[offset : offset + 2] == "P<":
            repaired = (
                line[offset : offset + 44]
                .ljust(44, "<")
            )

            if len(repaired) == 44:
                return (
                    repaired,
                    f"line1_prefix_trim_{offset}",
                )

    return line, None


def validation_dict(
    line_1: str,
    line_2: str,
) -> dict[str, Any]:
    return validate_td3_lines(
        line_1=line_1,
        line_2=line_2,
    ).to_dict()


def build_verified_profiles(
    records: list[dict[str, Any]],
) -> tuple[
    dict[str, Counter[str]],
    Counter[str],
]:
    """
    Learn weak document-format priors from checksum-verified MRZs.

    Nothing from GT is used:
    - passport number pattern by issuing country
    - frequency of nationality codes

    Example mask:
        Z6453675 -> A9999999
    """
    passport_patterns: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    nationality_frequency: Counter[str] = Counter()

    for record in records:
        selected = (
            record.get("selected_result")
            or {}
        )

        line_1 = clean_line(
            selected.get("line_1")
        )
        line_2 = clean_line(
            selected.get("line_2")
        )

        if (
            len(line_1) != 44
            or len(line_2) != 44
        ):
            continue

        validation = validation_dict(
            line_1,
            line_2,
        )

        if (
            validation.get(
                "all_main_checks_valid"
            )
            is not True
        ):
            continue

        issuer = issuer_from_line_1(
            line_1
        )

        number = passport_value(
            line_2
        )

        pattern = passport_pattern(
            number
        )

        nationality = (
            nationality_from_line_2(
                line_2
            )
        )

        if (
            len(issuer) == 3
            and issuer.isalpha()
            and number
            and "?" not in pattern
        ):
            passport_patterns[
                issuer
            ][pattern] += 1

        if (
            len(nationality) == 3
            and nationality.isalpha()
        ):
            nationality_frequency[
                nationality
            ] += 1

    return (
        dict(passport_patterns),
        nationality_frequency,
    )


def dominant_passport_pattern(
    issuer: str,
    profiles: dict[
        str,
        Counter[str],
    ],
    *,
    min_samples: int = 3,
    min_share: float = 0.60,
) -> str | None:
    counts = profiles.get(issuer)

    if not counts:
        return None

    total = sum(counts.values())

    if total < min_samples:
        return None

    pattern, count = counts.most_common(1)[0]

    if count / total < min_share:
        return None

    return pattern


def checksum_guided_passport_repair(
    line_1: str,
    line_2: str,
    profiles: dict[
        str,
        Counter[str],
    ],
) -> tuple[str, str | None]:
    """
    Repair at most ONE OCR-confusable character in passport number.

    Requirements:
    1. current passport checksum is false;
    2. candidate makes passport + final checksum true;
    3. country has a learned dominant passport-number pattern;
    4. exactly one checksum-valid candidate matches that pattern.

    This avoids choosing arbitrarily between several mathematical
    checksum solutions.
    """
    if (
        len(line_1) != 44
        or len(line_2) != 44
    ):
        return line_2, None

    current = validation_dict(
        line_1,
        line_2,
    )

    if (
        current.get(
            "passport_number_check_valid"
        )
        is not False
    ):
        return line_2, None

    issuer = issuer_from_line_1(
        line_1
    )

    dominant_pattern = (
        dominant_passport_pattern(
            issuer,
            profiles,
        )
    )

    if dominant_pattern is None:
        return line_2, None

    valid_candidates: list[
        tuple[str, int, str, str]
    ] = []

    for index in range(9):
        char = line_2[index]

        replacements = (
            PASSPORT_CONFUSIONS.get(
                char,
                (),
            )
        )

        for replacement in replacements:
            candidate = (
                line_2[:index]
                + replacement
                + line_2[index + 1 :]
            )

            validation = validation_dict(
                line_1,
                candidate,
            )

            if (
                validation.get(
                    "passport_number_check_valid"
                )
                is not True
                or validation.get(
                    "final_check_valid"
                )
                is not True
            ):
                continue

            value = passport_value(
                candidate
            )

            if (
                passport_pattern(value)
                != dominant_pattern
            ):
                continue

            valid_candidates.append(
                (
                    candidate,
                    index,
                    char,
                    replacement,
                )
            )

    unique_lines = {
        item[0]
        for item in valid_candidates
    }

    if len(unique_lines) != 1:
        return line_2, None

    candidate, index, old, new = (
        valid_candidates[0]
    )

    return (
        candidate,
        (
            "passport_checksum_pattern:"
            f"{index}:{old}>{new}"
        ),
    )


def hamming_distance(
    left: str,
    right: str,
) -> int:
    if len(left) != len(right):
        return 999

    return sum(
        a != b
        for a, b in zip(left, right)
    )


def recover_nationality_from_variants(
    line_1: str,
    line_2: str,
    ocr_record: dict[str, Any] | None,
    nationality_frequency: Counter[str],
) -> tuple[str, str | None]:
    """
    Field-level cross-variant recovery.

    Nationality is NOT protected by its own checksum. Therefore this
    repair is intentionally narrow:
    - selected MRZ must still fail main checks;
    - current nationality differs from issuing country;
    - another OCR variant yields a 3-letter nationality equal to issuer;
    - candidate differs by only one character;
    - candidate code has appeared in checksum-verified MRZ corpus.

    No country is hard-coded.
    """
    if (
        ocr_record is None
        or len(line_1) != 44
        or len(line_2) != 44
    ):
        return line_2, None

    current_validation = validation_dict(
        line_1,
        line_2,
    )

    if (
        current_validation.get(
            "all_main_checks_valid"
        )
        is True
    ):
        return line_2, None

    issuer = issuer_from_line_1(
        line_1
    )
    current_nat = nationality_from_line_2(
        line_2
    )

    if (
        len(issuer) != 3
        or not issuer.isalpha()
        or current_nat == issuer
    ):
        return line_2, None

    candidates: dict[
        str,
        list[str],
    ] = defaultdict(list)

    variants = (
        ocr_record.get("variants")
        or {}
    )

    for variant_name, variant in variants.items():
        variant_line_2 = clean_line(
            (variant or {}).get("line_2")
        )

        if len(variant_line_2) != 44:
            continue

        candidate_nat = (
            nationality_from_line_2(
                variant_line_2
            )
        )

        if (
            candidate_nat != issuer
            or hamming_distance(
                current_nat,
                candidate_nat,
            )
            > 1
            or nationality_frequency[
                candidate_nat
            ]
            <= 0
        ):
            continue

        raw_field = variant_line_2[10:13]

        candidates[
            raw_field
        ].append(
            str(variant_name)
        )

    if not candidates:
        return line_2, None

    # Prefer support from the most variants; require a unique winner.
    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            len(item[1]),
            nationality_frequency[
                alpha_code(item[0])
            ],
        ),
        reverse=True,
    )

    if (
        len(ranked) > 1
        and (
            len(ranked[0][1]),
            nationality_frequency[
                alpha_code(ranked[0][0])
            ],
        )
        == (
            len(ranked[1][1]),
            nationality_frequency[
                alpha_code(ranked[1][0])
            ],
        )
    ):
        return line_2, None

    raw_field, supporters = ranked[0]

    repaired = (
        line_2[:10]
        + raw_field
        + line_2[13:]
    )

    return (
        repaired,
        (
            "nationality_cross_variant:"
            + ",".join(supporters)
        ),
    )


def record_key(
    record: dict[str, Any],
) -> str:
    filename = str(
        record.get("filename")
        or ""
    )

    if filename:
        return filename

    sample_id = str(
        record.get("sample_id")
        or ""
    )

    return (
        f"{sample_id}.jpg"
        if sample_id
        else ""
    )


def parse_one_row(
    row: dict[str, str],
    *,
    ocr_record: dict[str, Any] | None,
    passport_profiles: dict[
        str,
        Counter[str],
    ],
    nationality_frequency: Counter[str],
) -> dict[str, Any]:
    filename = row.get(
        "filename",
        "",
    )

    original_line_1 = clean_line(
        row.get("selected_line_1", "")
    )
    original_line_2 = clean_line(
        row.get("selected_line_2", "")
    )

    repair_methods: list[str] = []

    line_1, method = (
        repair_td3_line_1_prefix(
            original_line_1
        )
    )

    if method:
        repair_methods.append(method)

    line_2, method = (
        checksum_guided_passport_repair(
            line_1,
            original_line_2,
            passport_profiles,
        )
    )

    if method:
        repair_methods.append(method)

    line_2, method = (
        recover_nationality_from_variants(
            line_1,
            line_2,
            ocr_record,
            nationality_frequency,
        )
    )

    if method:
        repair_methods.append(method)

    line_1_length = len(line_1)
    line_2_length = len(line_2)

    common_metadata = {
        "filename": filename,
        "ocr_selected_variant": row.get(
            "selected_variant"
        ),
        "ocr_mean_confidence": row.get(
            "selected_mean_confidence"
        ),
        "ocr_line_1_original": original_line_1,
        "ocr_line_2_original": original_line_2,
        "ocr_line_1": line_1,
        "ocr_line_2": line_2,
        "ocr_line_1_length": line_1_length,
        "ocr_line_2_length": line_2_length,
        "mrz_repair_applied": bool(
            repair_methods
        ),
        "mrz_repair_methods": " | ".join(
            repair_methods
        ),
    }

    if not line_1 or not line_2:
        return {
            **common_metadata,
            "parse_status": "not_parsed",
            "parse_mode": None,
            "parse_error": (
                "Thiếu một hoặc cả hai dòng MRZ."
            ),
        }

    exact_length = (
        line_1_length == 44
        and line_2_length == 44
    )

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
        **common_metadata,
        "parse_mode": parse_mode,
        "document_type": parsed_dict[
            "document_type"
        ],
        "issuing_country": parsed_dict[
            "issuing_country"
        ],
        "surname": parsed_dict[
            "surname"
        ],
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
        "sex": parsed_dict[
            "sex"
        ],
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

    input_rows = load_rows(
        INPUT_CSV
    )

    ocr_records = load_ocr_records(
        INPUT_JSON
    )

    record_map = {
        record_key(record): record
        for record in ocr_records
        if record_key(record)
    }

    (
        passport_profiles,
        nationality_frequency,
    ) = build_verified_profiles(
        ocr_records
    )

    parsed_rows = [
        parse_one_row(
            row,
            ocr_record=record_map.get(
                row.get("filename", "")
            ),
            passport_profiles=(
                passport_profiles
            ),
            nationality_frequency=(
                nationality_frequency
            ),
        )
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
        if row.get("parse_mode")
        == "strict_44_44"
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
        if row.get("parse_status")
        == "not_parsed"
    )

    parser_error_count = sum(
        1
        for row in parsed_rows
        if row.get("parse_status")
        == "error"
    )

    success_count = sum(
        1
        for row in parsed_rows
        if row.get("parse_status")
        == "success"
    )

    repaired_count = sum(
        bool(
            row.get(
                "mrz_repair_applied"
            )
        )
        for row in parsed_rows
    )

    repair_methods = Counter()

    for row in parsed_rows:
        for method in str(
            row.get("mrz_repair_methods")
            or ""
        ).split(" | "):
            if method:
                repair_methods[
                    method.split(":", 1)[0]
                ] += 1

    print("=" * 72)
    print("KẾT QUẢ PARSE MRZ TD3 — MRZ V5")
    print("=" * 72)

    print(
        f"Tổng record OCR             : {total}"
    )
    print(
        f"Parse thành công            : {success_count}"
    )
    print(
        f"Strict 44 + 44              : {strict_count}"
    )
    print(
        f"Parse mềm                   : {soft_count}"
    )
    print(
        f"Không đủ 2 dòng             : {not_parsed_count}"
    )
    print(
        f"Parser error                : {parser_error_count}"
    )
    print(
        f"Records được repair         : {repaired_count}"
    )

    if repair_methods:
        print("\nRepair methods:")
        for method, count in (
            repair_methods.most_common()
        ):
            print(
                f"  {method:<30}: {count}"
            )

    print("\nOutput:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
