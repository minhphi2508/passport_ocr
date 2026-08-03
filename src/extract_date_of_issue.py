from __future__ import annotations

import csv
import json
import re
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIZ_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "viz_ocr"
    / "viz_ocr_full.json"
)

MRZ_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "mrz_parsed"
    / "mrz_parsed_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "date_of_issue_hybrid_v3"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "date_of_issue_hybrid_v3_results.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "date_of_issue_hybrid_v3_results.json"
)


# ============================================================
# LABEL VOCAB
# ============================================================

ISSUE_LABELS = [
    "date of issue",
    "issue date",
    "date issued",
    "date of issuance",
    "iss date",
    "iss. date",
    "issuing date",

    "date de delivrance",
    "date de délivrance",
    "date d emission",
    "date d'émission",

    "fecha de expedicion",
    "fecha de expedición",
    "fecha de emision",
    "fecha de emisión",

    "data di rilascio",
    "data wydania",
    "ausstellungsdatum",
    "datum van afgifte",
    "isavimo data",
    "išdavimo data",
]

STRONG_LABEL_KEYWORDS = {
    "issue",
    "issued",
    "issuance",
    "iss",
    "delivrance",
    "expedicion",
    "emision",
    "rilascio",
    "wydania",
    "ausstellung",
    "afgifte",
    "isavimo",
    "isdavimo",
}


MONTH_ALIASES = {
    1: {"jan", "january", "janvier", "ene", "enero", "januar"},
    2: {"feb", "february", "fevrier", "febrero", "februar"},
    3: {"mar", "march", "mars", "marzo", "maerz"},
    4: {"apr", "april", "avr", "avril", "abril"},
    5: {"may", "mai", "mayo"},
    6: {"jun", "june", "juin", "junio", "juni"},
    7: {"jul", "july", "juillet", "julio", "juli"},
    8: {"aug", "august", "aout", "agosto"},
    9: {"sep", "sept", "september", "septembre", "septiembre"},
    10: {"oct", "october", "octobre", "octubre", "oktober"},
    11: {"nov", "november", "novembre", "noviembre"},
    12: {"dec", "december", "decembre", "diciembre", "dezember"},
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    text = str(text).lower().strip()

    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "á": "a",
        "à": "a",
        "â": "a",
        "ä": "a",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "ô": "o",
        "ö": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ñ": "n",
        "ç": "c",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"[^a-z0-9/.\-\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


NORMALIZED_LABELS = [
    normalize_text(label)
    for label in ISSUE_LABELS
]

MONTH_LOOKUP: dict[str, int] = {}

for month_number, aliases in MONTH_ALIASES.items():
    for alias in aliases:
        MONTH_LOOKUP[
            normalize_text(alias)
        ] = month_number


# ============================================================
# LABEL MATCHING
# ============================================================

def has_strong_issue_signal(
    text: str,
) -> bool:
    normalized = normalize_text(text)

    if len(normalized) < 5:
        return False

    tokens = set(
        normalized.split()
    )

    if "date" in tokens:
        for keyword in STRONG_LABEL_KEYWORDS:
            if (
                keyword in normalized
            ):
                return True

    # Cho phép "iss. date" đã mất dấu chấm.
    if (
        "iss" in normalized
        and "date" in normalized
    ):
        return True

    return False


def label_similarity(
    text: str,
) -> float:
    normalized = normalize_text(text)

    if not normalized:
        return 0.0

    # Loại ngay label quá ngắn kiểu P, F.
    if len(normalized) < 5:
        return 0.0

    best = 0.0

    for label in NORMALIZED_LABELS:
        if (
            label in normalized
            or normalized in label
        ):
            similarity = 1.0
        else:
            similarity = (
                SequenceMatcher(
                    None,
                    normalized,
                    label,
                )
                .ratio()
            )

        best = max(
            best,
            similarity,
        )

    if not has_strong_issue_signal(
        normalized
    ):
        best *= 0.45

    return best


# ============================================================
# DATE HELPERS
# ============================================================

def safe_iso_date(
    year: int,
    month: int,
    day: int,
) -> str | None:
    try:
        return date(
            year,
            month,
            day,
        ).isoformat()

    except ValueError:
        return None


def normalize_two_digit_year(
    year_2: int,
    birth_iso: str | None,
    expiry_iso: str | None,
) -> int:
    candidates = [
        1900 + year_2,
        2000 + year_2,
    ]

    birth = (
        date.fromisoformat(birth_iso)
        if birth_iso
        else None
    )

    expiry = (
        date.fromisoformat(expiry_iso)
        if expiry_iso
        else None
    )

    valid = []

    for year in candidates:
        if birth and year < birth.year:
            continue

        if expiry and year > expiry.year:
            continue

        valid.append(year)

    if valid:
        return max(valid)

    current_year = date.today().year

    return min(
        candidates,
        key=lambda value: abs(
            value - current_year
        ),
    )


def parse_numeric_date(
    day: int,
    month: int,
    year_text: str,
    birth_iso: str | None,
    expiry_iso: str | None,
) -> str | None:
    # Chỉ chấp nhận YY hoặc YYYY.
    if len(year_text) not in {
        2,
        4,
    }:
        return None

    year = int(year_text)

    if len(year_text) == 2:
        year = normalize_two_digit_year(
            year,
            birth_iso=birth_iso,
            expiry_iso=expiry_iso,
        )

    return safe_iso_date(
        year,
        month,
        day,
    )


def extract_dates_from_text(
    text: str,
    birth_iso: str | None,
    expiry_iso: str | None,
) -> list[dict[str, str]]:
    normalized = normalize_text(
        text
    )

    results: list[
        dict[str, str]
    ] = []

    # --------------------------------------------------------
    # DD/MM/YY or YYYY
    # DD-MM-YY
    # DD.MM.YYYY
    # --------------------------------------------------------

    for match in re.finditer(
        r"\b"
        r"(\d{1,2})"
        r"[./\-]"
        r"(\d{1,2})"
        r"[./\-]"
        r"(\d{2}|\d{4})"
        r"\b",
        normalized,
    ):
        iso = parse_numeric_date(
            int(match.group(1)),
            int(match.group(2)),
            match.group(3),
            birth_iso,
            expiry_iso,
        )

        if iso:
            results.append(
                {
                    "raw": match.group(0),
                    "iso": iso,
                }
            )

    # --------------------------------------------------------
    # DD MM YY / YYYY
    # --------------------------------------------------------

    for match in re.finditer(
        r"\b"
        r"(\d{1,2})"
        r"\s+"
        r"(\d{1,2})"
        r"\s+"
        r"(\d{2}|\d{4})"
        r"\b",
        normalized,
    ):
        iso = parse_numeric_date(
            int(match.group(1)),
            int(match.group(2)),
            match.group(3),
            birth_iso,
            expiry_iso,
        )

        if iso:
            results.append(
                {
                    "raw": match.group(0),
                    "iso": iso,
                }
            )

    # --------------------------------------------------------
    # Compact DDMMYY / DDMMYYYY
    # --------------------------------------------------------

    for match in re.finditer(
        r"(?<!\d)(\d{6}|\d{8})(?!\d)",
        normalized,
    ):
        value = match.group(1)

        if len(value) == 6:
            day = int(value[0:2])
            month = int(value[2:4])
            year_text = value[4:6]

        else:
            day = int(value[0:2])
            month = int(value[2:4])
            year_text = value[4:8]

        iso = parse_numeric_date(
            day,
            month,
            year_text,
            birth_iso,
            expiry_iso,
        )

        if iso:
            results.append(
                {
                    "raw": value,
                    "iso": iso,
                }
            )

    # --------------------------------------------------------
    # Cho chuỗi bị dính:
    # 2104201521042017
    #
    # Thử mọi đoạn 8 chữ số bên trong.
    # --------------------------------------------------------

    for long_number in re.findall(
        r"\d{9,}",
        normalized,
    ):
        for index in range(
            0,
            len(long_number) - 7,
        ):
            candidate = (
                long_number[
                    index:index + 8
                ]
            )

            day = int(
                candidate[0:2]
            )

            month = int(
                candidate[2:4]
            )

            year_text = (
                candidate[4:8]
            )

            iso = parse_numeric_date(
                day,
                month,
                year_text,
                birth_iso,
                expiry_iso,
            )

            if iso:
                results.append(
                    {
                        "raw": candidate,
                        "iso": iso,
                    }
                )

    # --------------------------------------------------------
    # DDMMMYYYY / DD MMM YYYY / YY
    # --------------------------------------------------------

    for match in re.finditer(
        r"\b"
        r"(\d{1,2})"
        r"\s*"
        r"([a-z]{3,10})"
        r"\s*"
        r"(\d{2}|\d{4})"
        r"\b",
        normalized,
    ):
        month_token = (
            match.group(2)
        )

        month = MONTH_LOOKUP.get(
            month_token
        )

        if month is None:
            continue

        iso = parse_numeric_date(
            int(match.group(1)),
            month,
            match.group(3),
            birth_iso,
            expiry_iso,
        )

        if iso:
            results.append(
                {
                    "raw": match.group(0),
                    "iso": iso,
                }
            )

    # --------------------------------------------------------
    # Bilingual month:
    # 29 Apr / Avr 1998
    # 14 Mar / Mars 2001
    # --------------------------------------------------------

    for match in re.finditer(
        r"\b"
        r"(\d{1,2})"
        r"\s+"
        r"([a-z]{3,10})"
        r"\s*/\s*"
        r"[a-z]{3,10}"
        r"\s+"
        r"(\d{2}|\d{4})"
        r"\b",
        normalized,
    ):
        month = MONTH_LOOKUP.get(
            match.group(2)
        )

        if month is None:
            continue

        iso = parse_numeric_date(
            int(match.group(1)),
            month,
            match.group(3),
            birth_iso,
            expiry_iso,
        )

        if iso:
            results.append(
                {
                    "raw": match.group(0),
                    "iso": iso,
                }
            )

    # Dedupe
    unique = {}

    for result in results:
        unique[
            result["iso"]
        ] = result

    return list(
        unique.values()
    )


# ============================================================
# OCR GROUPS
# ============================================================

def build_text_groups(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = []

    # Single box.
    for item in items:
        groups.append(
            {
                "text": item.get(
                    "text",
                    "",
                ),
                "items": [item],
            }
        )

    # Ghép 2 hoặc 3 box liên tiếp.
    for size in (2, 3):
        for index in range(
            len(items) - size + 1
        ):
            chunk = items[
                index:index + size
            ]

            groups.append(
                {
                    "text": " ".join(
                        str(
                            item.get(
                                "text",
                                "",
                            )
                        )
                        for item in chunk
                    ),
                    "items": chunk,
                }
            )

    return groups


# ============================================================
# TEMPORAL RULES
# ============================================================

def hard_temporal_filter(
    candidate_iso: str,
    birth_iso: str | None,
    expiry_iso: str | None,
) -> bool:
    """
    True = candidate được phép.

    Khi có MRZ:
        DOB < DOI < EXP
    """
    candidate = date.fromisoformat(
        candidate_iso
    )

    if birth_iso:
        birth = date.fromisoformat(
            birth_iso
        )

        if candidate <= birth:
            return False

    if expiry_iso:
        expiry = date.fromisoformat(
            expiry_iso
        )

        if candidate >= expiry:
            return False

    return True


def temporal_score(
    candidate_iso: str,
    birth_iso: str | None,
    expiry_iso: str | None,
) -> float:
    if not hard_temporal_filter(
        candidate_iso,
        birth_iso,
        expiry_iso,
    ):
        return -1000.0

    score = 0.0

    if birth_iso:
        score += 10.0

    if expiry_iso:
        score += 20.0

        candidate = date.fromisoformat(
            candidate_iso
        )

        expiry = date.fromisoformat(
            expiry_iso
        )

        years = (
            (expiry - candidate).days
            / 365.25
        )

        if 0.5 <= years <= 11.5:
            score += 15.0

        if abs(years - 5) <= 1:
            score += 10.0

        if abs(years - 10) <= 1:
            score += 10.0

    return score


# ============================================================
# LOAD MRZ
# ============================================================

def load_mrz_rows(
) -> dict[str, dict[str, str]]:
    if not MRZ_CSV.exists():
        return {}

    with MRZ_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    return {
        row["filename"]: row
        for row in rows
        if row.get("filename")
    }


# ============================================================
# EXTRACT ONE
# ============================================================

def extract_one(
    record: dict[str, Any],
    mrz_map: dict[str, dict[str, str]],
) -> dict[str, Any]:
    filename = record.get(
        "filename"
    )

    selected_result = record.get(
        "selected_result",
        {},
    )

    items = selected_result.get(
        "items",
        [],
    )

    mrz = mrz_map.get(
        filename,
        {},
    )

    birth_iso = (
        mrz.get("birth_date")
        or None
    )

    expiry_iso = (
        mrz.get("expiry_date")
        or None
    )

    if not items:
        return {
            "filename": filename,
            "status": "no_ocr_items",
            "date_of_issue": None,
            "method": None,
            "score": None,
            "birth_date_mrz": birth_iso,
            "expiry_date_mrz": expiry_iso,
        }

    groups = build_text_groups(
        items
    )

    date_candidates = []

    for group in groups:
        dates = extract_dates_from_text(
            group["text"],
            birth_iso=birth_iso,
            expiry_iso=expiry_iso,
        )

        for date_candidate in dates:
            date_candidates.append(
                {
                    "date": date_candidate,
                    "group": group,
                }
            )

    # --------------------------------------------------------
    # Hard MRZ filter
    # --------------------------------------------------------

    filtered_candidates = []

    for candidate in date_candidates:
        iso = candidate[
            "date"
        ]["iso"]

        if (
            birth_iso
            or expiry_iso
        ):
            if not hard_temporal_filter(
                iso,
                birth_iso,
                expiry_iso,
            ):
                continue

        filtered_candidates.append(
            candidate
        )

    # --------------------------------------------------------
    # Strong DOI labels only
    # --------------------------------------------------------

    label_candidates = []

    for item in items:
        text = item.get(
            "text",
            "",
        )

        similarity = (
            label_similarity(text)
        )

        if (
            similarity >= 0.60
            and has_strong_issue_signal(
                text
            )
        ):
            label_candidates.append(
                (
                    similarity,
                    item,
                )
            )

    scored = []

    # --------------------------------------------------------
    # LEVEL 1: STRONG LABEL
    # --------------------------------------------------------

    for similarity, label_item in label_candidates:
        label_index = items.index(
            label_item
        )

        for candidate in filtered_candidates:
            group_items = candidate[
                "group"
            ]["items"]

            indices = [
                items.index(item)
                for item in group_items
                if item in items
            ]

            if not indices:
                continue

            distance = min(
                abs(
                    index
                    - label_index
                )
                for index in indices
            )

            # Ưu tiên các OCR boxes gần label.
            position_score = max(
                0.0,
                40.0
                - distance * 8.0,
            )

            iso = candidate[
                "date"
            ]["iso"]

            score = (
                similarity * 40.0
                + position_score
                + temporal_score(
                    iso,
                    birth_iso,
                    expiry_iso,
                )
            )

            scored.append(
                {
                    "iso": iso,
                    "raw": candidate[
                        "date"
                    ]["raw"],
                    "score": score,
                    "method": "strong_label_mrz",
                    "label_text": (
                        label_item.get(
                            "text"
                        )
                    ),
                    "candidate_text": (
                        candidate[
                            "group"
                        ]["text"]
                    ),
                }
            )

    # --------------------------------------------------------
    # LEVEL 2: MRZ ASSISTED
    # --------------------------------------------------------

    if (
        not scored
        and (
            birth_iso
            or expiry_iso
        )
    ):
        for candidate in filtered_candidates:
            iso = candidate[
                "date"
            ]["iso"]

            score = (
                30.0
                + temporal_score(
                    iso,
                    birth_iso,
                    expiry_iso,
                )
            )

            scored.append(
                {
                    "iso": iso,
                    "raw": candidate[
                        "date"
                    ]["raw"],
                    "score": score,
                    "method": "mrz_assisted",
                    "label_text": None,
                    "candidate_text": (
                        candidate[
                            "group"
                        ]["text"]
                    ),
                }
            )

    # --------------------------------------------------------
    # LEVEL 3:
    # Không có MRZ thì chỉ cho phép strong label.
    # Không generic guessing.
    # --------------------------------------------------------

    if not scored:
        return {
            "filename": filename,
            "status": "no_date_found",
            "date_of_issue": None,
            "method": None,
            "score": None,
            "birth_date_mrz": birth_iso,
            "expiry_date_mrz": expiry_iso,
            "all_candidate_dates": list(
                dict.fromkeys(
                    candidate[
                        "date"
                    ]["iso"]
                    for candidate
                    in date_candidates
                )
            ),
        }

    # --------------------------------------------------------
    # DEDUPE
    # --------------------------------------------------------

    best_by_iso = {}

    for candidate in scored:
        iso = candidate["iso"]

        if (
            iso not in best_by_iso
            or candidate["score"]
            > best_by_iso[iso]["score"]
        ):
            best_by_iso[
                iso
            ] = candidate

    ranked = sorted(
        best_by_iso.values(),
        key=lambda value: value[
            "score"
        ],
        reverse=True,
    )

    best = ranked[0]

    second_score = (
        ranked[1]["score"]
        if len(ranked) > 1
        else None
    )

    margin = (
        best["score"]
        - second_score
        if second_score is not None
        else None
    )

    if (
        best["method"]
        == "strong_label_mrz"
        and best["score"] >= 75
        and (
            margin is None
            or margin >= 5
        )
    ):
        status = "high_confidence"

    elif best["score"] >= 55:
        status = "medium_confidence"

    else:
        status = "low_confidence"

    return {
        "filename": filename,
        "status": status,
        "date_of_issue": best[
            "iso"
        ],
        "raw_date": best[
            "raw"
        ],
        "method": best[
            "method"
        ],
        "score": round(
            float(best["score"]),
            3,
        ),
        "score_margin": (
            round(
                float(margin),
                3,
            )
            if margin is not None
            else None
        ),
        "birth_date_mrz": birth_iso,
        "expiry_date_mrz": expiry_iso,
        "label_text": best[
            "label_text"
        ],
        "candidate_text": best[
            "candidate_text"
        ],
        "all_candidate_dates": [
            item["iso"]
            for item in ranked
        ],
    }


# ============================================================
# IO
# ============================================================

def load_viz_records(
) -> list[dict[str, Any]]:
    if not VIZ_JSON.exists():
        raise FileNotFoundError(
            VIZ_JSON
        )

    with VIZ_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "viz_ocr_full.json không phải list."
        )

    return data


def write_outputs(
    results: list[dict[str, Any]],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    rows = []

    for result in results:
        row = dict(result)

        candidate_dates = row.get(
            "all_candidate_dates"
        )

        if isinstance(
            candidate_dates,
            list,
        ):
            row[
                "all_candidate_dates"
            ] = " | ".join(
                candidate_dates
            )

        rows.append(
            row
        )

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    viz_records = (
        load_viz_records()
    )

    mrz_map = (
        load_mrz_rows()
    )

    results = [
        extract_one(
            record,
            mrz_map,
        )
        for record in viz_records
    ]

    write_outputs(
        results
    )

    status_counts = {}
    method_counts = {}

    extracted = 0

    for result in results:
        status = result[
            "status"
        ]

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

        method = result.get(
            "method"
        )

        if method:
            method_counts[
                method
            ] = (
                method_counts.get(
                    method,
                    0,
                )
                + 1
            )

        if result.get(
            "date_of_issue"
        ):
            extracted += 1

    print("=" * 76)
    print("KẾT QUẢ DOI HYBRID V3")
    print("=" * 76)

    print(
        f"Tổng VIZ OCR          : "
        f"{len(results)}"
    )

    print(
        f"Extract được DOI      : "
        f"{extracted}"
    )

    print("\nStatus:")

    for status, count in sorted(
        status_counts.items()
    ):
        print(
            f"{status:<28}: "
            f"{count}"
        )

    print("\nMethod:")

    for method, count in sorted(
        method_counts.items()
    ):
        print(
            f"{method:<28}: "
            f"{count}"
        )

    print("\nCác case không high-confidence:")

    for result in results:
        if (
            result["status"]
            == "high_confidence"
        ):
            continue

        print()
        print(
            result[
                "filename"
            ]
        )

        print(
            "Status :",
            result[
                "status"
            ],
        )

        print(
            "DOI    :",
            result.get(
                "date_of_issue"
            ),
        )

        print(
            "Method :",
            result.get(
                "method"
            ),
        )

        print(
            "DOB    :",
            result.get(
                "birth_date_mrz"
            ),
        )

        print(
            "Expiry :",
            result.get(
                "expiry_date_mrz"
            ),
        )

        print(
            "Label  :",
            result.get(
                "label_text"
            ),
        )

        print(
            "Text   :",
            result.get(
                "candidate_text"
            ),
        )

    print("\nCSV:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()