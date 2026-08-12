from __future__ import annotations

import csv
import json
import re
import unicodedata

from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

VIZ_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "viz_ocr"
    / "viz_ocr_full.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "viz_fields"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "viz_fields_results.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "viz_fields_results.json"
)


# ============================================================
# CONFIG
# ============================================================

# Label similarity tối thiểu để coi một OCR item
# là label của field.
LABEL_SIMILARITY_THRESHOLD = 0.58

# Không dùng OCR item confidence quá thấp
# làm value trừ khi không có lựa chọn tốt hơn.
MIN_VALUE_CONFIDENCE = 0.35

# Candidate ở quá xa label theo chiều dọc
# sẽ bị loại.
MAX_VERTICAL_DISTANCE_MULTIPLIER = 4.2

# Cho phép value nằm bên phải label
# trong cùng một hàng.
MAX_HORIZONTAL_GAP_MULTIPLIER = 3.5


# ============================================================
# FIELD LABELS
# ============================================================

FIELD_LABELS: dict[str, list[str]] = {
    "passport_number": [
        "passport no",
        "passport no.",
        "passport number",
        "document no",
        "document number",
        "numero de passeport",
        "no de passeport",
        "numero de pasaporte",
        "passport nr",
    ],

    "surname": [
        "surname",
        "family name",
        "last name",
        "nom",
        "apellido",
        "apellidos",
    ],

    "given_names": [
        "given name",
        "given names",
        "given name(s)",
        "first name",
        "first names",
        "prenoms",
        "prenom",
        "nombre",
        "nombres",
    ],

    "nationality": [
        "nationality",
        "nationalite",
        "nacionalidad",
        "citizenship",
    ],

    "date_of_birth": [
        "date of birth",
        "birth date",
        "date de naissance",
        "fecha de nacimiento",
        "datum urodzenia",
        "geburtsdatum",
    ],

    "sex": [
        "sex",
        "gender",
        "sexe",
        "sexo",
        "geschlecht",
    ],

    "date_of_expiry": [
        "date of expiry",
        "expiry date",
        "date of expiration",
        "expiration date",
        "date d expiration",
        "date dexpiration",
        "fecha de caducidad",
        "fecha de vencimiento",
        "valid until",
        "valid thru",
    ],
}


# ============================================================
# MONTHS
# ============================================================

MONTHS = {
    "jan": 1,
    "january": 1,
    "janvier": 1,

    "feb": 2,
    "february": 2,
    "fevrier": 2,

    "mar": 3,
    "march": 3,
    "mars": 3,

    "apr": 4,
    "april": 4,
    "avr": 4,
    "avril": 4,

    "may": 5,
    "mai": 5,

    "jun": 6,
    "june": 6,
    "juin": 6,

    "jul": 7,
    "july": 7,
    "juillet": 7,

    "aug": 8,
    "august": 8,
    "aout": 8,

    "sep": 9,
    "sept": 9,
    "september": 9,
    "septembre": 9,

    "oct": 10,
    "october": 10,
    "octobre": 10,

    "nov": 11,
    "november": 11,
    "novembre": 11,

    "dec": 12,
    "december": 12,
    "decembre": 12,
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def strip_accents(
    text: str,
) -> str:

    normalized = unicodedata.normalize(
        "NFKD",
        str(text),
    )

    return "".join(
        char
        for char in normalized
        if not unicodedata.combining(
            char
        )
    )


def normalize_text(
    text: str,
) -> str:

    text = strip_accents(
        text
    ).lower()

    text = text.replace(
        "0",
        "o",
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_value_text(
    text: str,
) -> str:

    text = str(
        text
    ).strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip(
        " |/:;,-"
    )


# ============================================================
# GEOMETRY
# ============================================================

def get_box(
    item: dict[str, Any],
) -> tuple[
    float,
    float,
    float,
    float,
] | None:

    box = item.get(
        "box"
    )

    if (
        not isinstance(
            box,
            list,
        )
        or len(box) < 4
    ):
        return None

    try:

        x1 = float(
            box[0]
        )

        y1 = float(
            box[1]
        )

        x2 = float(
            box[2]
        )

        y2 = float(
            box[3]
        )

        return (
            x1,
            y1,
            x2,
            y2,
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def box_width(
    box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:

    return max(
        1.0,
        box[2] - box[0],
    )


def box_height(
    box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:

    return max(
        1.0,
        box[3] - box[1],
    )


def center_x(
    box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:

    return (
        box[0]
        + box[2]
    ) / 2.0


def center_y(
    box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:

    return (
        box[1]
        + box[3]
    ) / 2.0


def horizontal_overlap_ratio(
    first: tuple[
        float,
        float,
        float,
        float,
    ],
    second: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:

    overlap = max(
        0.0,
        min(
            first[2],
            second[2],
        )
        - max(
            first[0],
            second[0],
        ),
    )

    denominator = min(
        box_width(first),
        box_width(second),
    )

    return (
        overlap
        / denominator
        if denominator > 0
        else 0.0
    )


# ============================================================
# LABEL MATCHING
# ============================================================

def label_similarity(
    text: str,
    target_label: str,
) -> float:

    normalized_text = (
        normalize_text(
            text
        )
    )

    normalized_label = (
        normalize_text(
            target_label
        )
    )

    if (
        not normalized_text
        or not normalized_label
    ):
        return 0.0

    # Exact containment là evidence mạnh.
    if (
        normalized_label
        in normalized_text
    ):
        return 1.0

    if (
        normalized_text
        in normalized_label
        and len(
            normalized_text
        ) >= 4
    ):
        return 0.95

    similarity = (
        SequenceMatcher(
            None,
            normalized_text,
            normalized_label,
        ).ratio()
    )

    # Token-based score giúp các OCR string
    # có phần ngôn ngữ địa phương ở trước:
    #
    # "abc / Nationality"
    text_tokens = set(
        normalized_text.split()
    )

    label_tokens = set(
        normalized_label.split()
    )

    if label_tokens:

        token_overlap = (
            len(
                text_tokens
                & label_tokens
            )
            / len(
                label_tokens
            )
        )

        similarity = max(
            similarity,
            token_overlap,
        )

    return similarity


def best_label_match(
    text: str,
    field_name: str,
) -> float:

    return max(
        label_similarity(
            text,
            label,
        )
        for label in FIELD_LABELS[
            field_name
        ]
    )


def item_is_any_label(
    item: dict[str, Any],
) -> bool:

    text = item.get(
        "text",
        "",
    )

    for field_name in FIELD_LABELS:

        if (
            best_label_match(
                text,
                field_name,
            )
            >= LABEL_SIMILARITY_THRESHOLD
        ):
            return True

    return False


# ============================================================
# VALUE VALIDATION
# ============================================================

def parse_date_text(
    value: str,
) -> str | None:

    raw = strip_accents(
        value
    ).strip()

    # --------------------------------------------------------
    # DD/MM/YYYY etc.
    # --------------------------------------------------------

    match = re.search(
        r"\b"
        r"(\d{1,2})"
        r"[./\-]"
        r"(\d{1,2})"
        r"[./\-]"
        r"(\d{2,4})"
        r"\b",
        raw,
    )

    if match:

        day = int(
            match.group(1)
        )

        month = int(
            match.group(2)
        )

        year_text = (
            match.group(3)
        )

        year = int(
            year_text
        )

        if len(
            year_text
        ) == 2:

            # Passport date:
            # heuristic đơn giản chỉ để normalize VIZ.
            year = (
                2000 + year
                if year <= 40
                else 1900 + year
            )

        try:

            return date(
                year,
                month,
                day,
            ).isoformat()

        except ValueError:
            return None

    # --------------------------------------------------------
    # DD MMM YYYY
    # --------------------------------------------------------

    normalized = normalize_text(
        raw
    )

    match = re.search(
        r"\b"
        r"(\d{1,2})"
        r"\s+"
        r"([a-z]{3,12})"
        r"\s+"
        r"(\d{2,4})"
        r"\b",
        normalized,
    )

    if match:

        day = int(
            match.group(1)
        )

        month_token = (
            match.group(2)
        )

        month = MONTHS.get(
            month_token
        )

        if month is None:

            # Cho phép OCR month dài/rác nhẹ.
            best_month = None
            best_score = 0.0

            for (
                alias,
                month_number,
            ) in MONTHS.items():

                score = (
                    SequenceMatcher(
                        None,
                        month_token,
                        alias,
                    ).ratio()
                )

                if (
                    score
                    > best_score
                ):

                    best_score = score
                    best_month = (
                        month_number
                    )

            if (
                best_score
                >= 0.72
            ):
                month = (
                    best_month
                )

        if month is None:
            return None

        year_text = (
            match.group(3)
        )

        year = int(
            year_text
        )

        if (
            len(
                year_text
            )
            == 2
        ):

            year = (
                2000 + year
                if year <= 40
                else 1900 + year
            )

        try:

            return date(
                year,
                month,
                day,
            ).isoformat()

        except ValueError:
            return None

    return None


def normalize_sex(
    value: str,
) -> str | None:

    normalized = (
        normalize_text(
            value
        )
    )

    if normalized in {
        "m",
        "male",
    }:
        return "M"

    if normalized in {
        "f",
        "female",
    }:
        return "F"

    if normalized in {
        "x",
    }:
        return "X"

    return None


def valid_passport_number(
    value: str,
) -> bool:

    compact = re.sub(
        r"[^A-Za-z0-9]",
        "",
        value,
    ).upper()

    if not (
        5
        <= len(compact)
        <= 12
    ):
        return False

    # Ít nhất phải chứa digit.
    if not re.search(
        r"\d",
        compact,
    ):
        return False

    return True


def normalize_passport_number(
    value: str,
) -> str | None:

    compact = re.sub(
        r"[^A-Za-z0-9]",
        "",
        value,
    ).upper()

    if not valid_passport_number(
        compact
    ):
        return None

    return compact


def valid_name(
    value: str,
) -> bool:

    normalized = (
        normalize_value_text(
            value
        )
    )

    if len(
        normalized
    ) < 2:
        return False

    # Name không nên là date.
    if parse_date_text(
        normalized
    ):
        return False

    alpha_count = sum(
        char.isalpha()
        for char in normalized
    )

    return (
        alpha_count
        >= 2
    )


def valid_nationality(
    value: str,
) -> bool:

    normalized = (
        normalize_value_text(
            value
        )
    )

    if len(
        normalized
    ) < 2:
        return False

    if parse_date_text(
        normalized
    ):
        return False

    return any(
        char.isalpha()
        for char in normalized
    )


def normalize_candidate_value(
    field_name: str,
    value: str,
) -> str | None:

    cleaned = normalize_value_text(
        value
    )

    if not cleaned:
        return None

    if (
        field_name
        == "passport_number"
    ):

        return (
            normalize_passport_number(
                cleaned
            )
        )

    if field_name in {
        "date_of_birth",
        "date_of_expiry",
    }:

        return parse_date_text(
            cleaned
        )

    if (
        field_name
        == "sex"
    ):

        return normalize_sex(
            cleaned
        )

    if field_name in {
        "surname",
        "given_names",
    }:

        if not valid_name(
            cleaned
        ):
            return None

        return cleaned.upper()

    if (
        field_name
        == "nationality"
    ):

        if not valid_nationality(
            cleaned
        ):
            return None

        return cleaned.upper()

    return cleaned


# ============================================================
# LABEL VALUE CANDIDATES
# ============================================================

def spatial_candidate_score(
    label_item: dict[str, Any],
    value_item: dict[str, Any],
) -> tuple[
    float,
    str,
]:

    label_box = get_box(
        label_item
    )

    value_box = get_box(
        value_item
    )

    if (
        label_box is None
        or value_box is None
    ):
        return (
            -1000.0,
            "missing_geometry",
        )

    label_height = (
        box_height(
            label_box
        )
    )

    label_width = (
        box_width(
            label_box
        )
    )

    vertical_gap = (
        value_box[1]
        - label_box[3]
    )

    horizontal_gap = (
        value_box[0]
        - label_box[2]
    )

    overlap = (
        horizontal_overlap_ratio(
            label_box,
            value_box,
        )
    )

    # --------------------------------------------------------
    # MODE 1:
    # value nằm bên dưới label.
    # --------------------------------------------------------

    if (
        vertical_gap
        >= -0.35
        * label_height
        and vertical_gap
        <= MAX_VERTICAL_DISTANCE_MULTIPLIER
        * label_height
    ):

        x_distance = abs(
            center_x(
                value_box
            )
            - center_x(
                label_box
            )
        )

        x_scale = max(
            label_width,
            box_width(
                value_box
            ),
            1.0,
        )

        x_penalty = (
            x_distance
            / x_scale
        )

        y_penalty = (
            max(
                0.0,
                vertical_gap,
            )
            / label_height
        )

        score = 4.0

        score += (
            overlap * 3.0
        )

        score -= (
            y_penalty * 0.45
        )

        score -= (
            x_penalty * 0.90
        )

        return (
            score,
            "below_label",
        )

    # --------------------------------------------------------
    # MODE 2:
    # value nằm bên phải label cùng hàng.
    # --------------------------------------------------------

    vertical_center_delta = abs(
        center_y(
            value_box
        )
        - center_y(
            label_box
        )
    )

    if (
        horizontal_gap >= 0
        and horizontal_gap
        <= MAX_HORIZONTAL_GAP_MULTIPLIER
        * label_height
        and vertical_center_delta
        <= 1.0
        * max(
            label_height,
            box_height(
                value_box
            ),
        )
    ):

        score = 3.3

        score -= (
            horizontal_gap
            / max(
                label_height,
                1.0,
            )
            * 0.25
        )

        return (
            score,
            "right_of_label",
        )

    return (
        -1000.0,
        "not_spatially_related",
    )


# ============================================================
# SAME ITEM VALUE
#
# VD:
# "Passport No. A1234567"
# ============================================================

def extract_inline_value(
    item_text: str,
    field_name: str,
) -> str | None:

    original = (
        normalize_value_text(
            item_text
        )
    )

    if not original:
        return None

    for label in FIELD_LABELS[
        field_name
    ]:

        normalized_item = (
            normalize_text(
                original
            )
        )

        normalized_label = (
            normalize_text(
                label
            )
        )

        if (
            normalized_label
            not in normalized_item
        ):
            continue

        # Inline extraction ở mức conservative:
        # chỉ áp dụng cho các field có format mạnh.
        if (
            field_name
            == "passport_number"
        ):

            candidates = re.findall(
                r"\b[A-Z0-9]{5,12}\b",
                original.upper(),
            )

            for candidate in candidates:

                normalized_value = (
                    normalize_passport_number(
                        candidate
                    )
                )

                if normalized_value:
                    return normalized_value

        if field_name in {
            "date_of_birth",
            "date_of_expiry",
        }:

            parsed = parse_date_text(
                original
            )

            if parsed:
                return parsed

        if (
            field_name
            == "sex"
        ):

            match = re.search(
                r"\b[MFx]\b",
                original,
                flags=re.IGNORECASE,
            )

            if match:

                return (
                    normalize_sex(
                        match.group(0)
                    )
                )

    return None


# ============================================================
# EXTRACT FIELD FROM ONE VARIANT
# ============================================================

def extract_field_from_variant(
    items: list[
        dict[
            str,
            Any,
        ]
    ],
    field_name: str,
    variant_name: str,
) -> dict[str, Any] | None:

    candidates = []

    # ========================================================
    # FIND LABELS
    # ========================================================

    label_items = []

    for (
        index,
        item,
    ) in enumerate(
        items
    ):

        text = str(
            item.get(
                "text",
                "",
            )
        )

        similarity = (
            best_label_match(
                text,
                field_name,
            )
        )

        if (
            similarity
            < LABEL_SIMILARITY_THRESHOLD
        ):
            continue

        label_items.append(
            (
                index,
                item,
                similarity,
            )
        )

        # Inline value.
        inline = (
            extract_inline_value(
                text,
                field_name,
            )
        )

        if (
            inline
            is not None
        ):

            confidence = item.get(
                "confidence"
            )

            try:

                confidence = (
                    float(
                        confidence
                    )
                    if confidence
                    is not None
                    else 0.0
                )

            except (
                TypeError,
                ValueError,
            ):

                confidence = 0.0

            candidates.append(
                {
                    "value":
                        inline,

                    "raw_text":
                        text,

                    "label_text":
                        text,

                    "method":
                        "inline",

                    "variant":
                        variant_name,

                    "label_similarity":
                        similarity,

                    "ocr_confidence":
                        confidence,

                    "score":
                        (
                            7.0
                            + similarity * 2.0
                            + confidence
                        ),
                }
            )

    # ========================================================
    # LABEL → NEARBY VALUE
    # ========================================================

    for (
        label_index,
        label_item,
        label_similarity_score,
    ) in label_items:

        for (
            value_index,
            value_item,
        ) in enumerate(
            items
        ):

            if (
                value_index
                == label_index
            ):
                continue

            # Không lấy một label khác làm value.
            if item_is_any_label(
                value_item
            ):
                continue

            raw_value = str(
                value_item.get(
                    "text",
                    "",
                )
            ).strip()

            if not raw_value:
                continue

            normalized_value = (
                normalize_candidate_value(
                    field_name,
                    raw_value,
                )
            )

            if (
                normalized_value
                is None
            ):
                continue

            (
                spatial_score,
                spatial_method,
            ) = spatial_candidate_score(
                label_item,
                value_item,
            )

            if (
                spatial_score
                <= -999
            ):
                continue

            confidence = (
                value_item.get(
                    "confidence"
                )
            )

            try:

                confidence = (
                    float(
                        confidence
                    )
                    if confidence
                    is not None
                    else 0.0
                )

            except (
                TypeError,
                ValueError,
            ):

                confidence = 0.0

            # Confidence thấp không bị hard reject,
            # chỉ bị penalty.
            confidence_score = max(
                0.0,
                confidence,
            )

            if (
                confidence
                < MIN_VALUE_CONFIDENCE
            ):

                confidence_score -= 0.8

            # OCR items gần nhau trong list thường cũng
            # là evidence hữu ích.
            sequence_distance = abs(
                value_index
                - label_index
            )

            sequence_bonus = max(
                0.0,
                1.5
                - 0.20
                * sequence_distance,
            )

            total_score = (
                spatial_score
                + (
                    label_similarity_score
                    * 2.3
                )
                + confidence_score
                + sequence_bonus
            )

            candidates.append(
                {
                    "value":
                        normalized_value,

                    "raw_text":
                        raw_value,

                    "label_text":
                        label_item.get(
                            "text",
                            "",
                        ),

                    "method":
                        spatial_method,

                    "variant":
                        variant_name,

                    "label_similarity":
                        label_similarity_score,

                    "ocr_confidence":
                        confidence,

                    "score":
                        total_score,
                }
            )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda candidate:
            candidate[
                "score"
            ],
    )


# ============================================================
# CROSS-VARIANT CONSENSUS
# ============================================================

def choose_cross_variant_result(
    results:
        list[
            dict[
                str,
                Any,
            ]
        ],
) -> dict[str, Any] | None:

    if not results:
        return None

    grouped: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    for result in results:

        value = str(
            result[
                "value"
            ]
        ).strip()

        grouped.setdefault(
            value,
            [],
        ).append(
            result
        )

    best_value = None
    best_group_score = (
        -1e9
    )

    for (
        value,
        group,
    ) in grouped.items():

        best_single = max(
            item[
                "score"
            ]
            for item in group
        )

        # Bonus nếu nhiều OCR variants đồng ý.
        consensus_bonus = (
            max(
                0,
                len(group) - 1,
            )
            * 1.4
        )

        group_score = (
            best_single
            + consensus_bonus
        )

        if (
            group_score
            > best_group_score
        ):

            best_group_score = (
                group_score
            )

            best_value = (
                value
            )

    assert (
        best_value
        is not None
    )

    winning_group = grouped[
        best_value
    ]

    best_result = max(
        winning_group,
        key=lambda item:
            item[
                "score"
            ],
    ).copy()

    best_result[
        "variant_agreement"
    ] = len(
        winning_group
    )

    best_result[
        "final_score"
    ] = (
        best_group_score
    )

    return best_result


# ============================================================
# PROCESS ONE OCR RECORD
# ============================================================

def process_record(
    record: dict[str, Any],
) -> dict[str, Any]:

    filename = record.get(
        "filename"
    )

    variant_records = (
        record.get(
            "variants"
        )
        or {}
    )

    # Đảm bảo selected_result vẫn được dùng
    # nếu JSON nào đó thiếu variants.
    if (
        not variant_records
        and record.get(
            "selected_result"
        )
    ):

        selected_name = (
            record.get(
                "selected_variant"
            )
            or "selected"
        )

        variant_records = {
            selected_name:
                record[
                    "selected_result"
                ]
        }

    field_results: dict[
        str,
        Any,
    ] = {}

    for field_name in (
        "passport_number",
        "surname",
        "given_names",
        "nationality",
        "date_of_birth",
        "sex",
        "date_of_expiry",
    ):

        variant_candidates = []

        for (
            variant_name,
            variant_result,
        ) in (
            variant_records.items()
        ):

            if (
                not isinstance(
                    variant_result,
                    dict,
                )
            ):
                continue

            if (
                variant_result.get(
                    "status"
                )
                != "success"
            ):
                continue

            items = (
                variant_result.get(
                    "items"
                )
                or []
            )

            if not items:
                continue

            candidate = (
                extract_field_from_variant(
                    items=items,
                    field_name=field_name,
                    variant_name=(
                        variant_name
                    ),
                )
            )

            if (
                candidate
                is not None
            ):

                variant_candidates.append(
                    candidate
                )

        field_results[
            field_name
        ] = (
            choose_cross_variant_result(
                variant_candidates
            )
        )

    output = {
        "filename":
            filename,

        "viz_selected_variant":
            record.get(
                "selected_variant"
            ),

        "viz_fallback_used":
            record.get(
                "fallback_used"
            ),
    }

    extracted_count = 0

    for (
        field_name,
        result,
    ) in field_results.items():

        if result is None:

            output[
                field_name
            ] = None

            output[
                f"{field_name}_metadata"
            ] = {
                "status":
                    "not_found",

                "method":
                    None,

                "variant":
                    None,

                "score":
                    None,

                "variant_agreement":
                    0,

                "label_text":
                    None,

                "raw_text":
                    None,

                "ocr_confidence":
                    None,
            }

            continue

        extracted_count += 1

        output[
            field_name
        ] = result[
            "value"
        ]

        output[
            f"{field_name}_metadata"
        ] = {
            "status":
                "found",

            "method":
                result.get(
                    "method"
                ),

            "variant":
                result.get(
                    "variant"
                ),

            "score":
                round(
                    float(
                        result.get(
                            "final_score",
                            result.get(
                                "score",
                                0.0,
                            ),
                        )
                    ),
                    4,
                ),

            "variant_agreement":
                result.get(
                    "variant_agreement",
                    1,
                ),

            "label_text":
                result.get(
                    "label_text"
                ),

            "raw_text":
                result.get(
                    "raw_text"
                ),

            "ocr_confidence":
                result.get(
                    "ocr_confidence"
                ),
        }

    output[
        "extracted_field_count"
    ] = extracted_count

    if (
        extracted_count
        >= 5
    ):

        output[
            "viz_extraction_status"
        ] = "strong"

    elif (
        extracted_count
        >= 2
    ):

        output[
            "viz_extraction_status"
        ] = "partial"

    elif (
        extracted_count
        == 1
    ):

        output[
            "viz_extraction_status"
        ] = "weak"

    else:

        output[
            "viz_extraction_status"
        ] = "no_fields"

    return output


# ============================================================
# FLATTEN CSV
# ============================================================

def flatten_for_csv(
    record: dict[str, Any],
) -> dict[str, Any]:

    row = {
        "filename":
            record[
                "filename"
            ],

        "viz_extraction_status":
            record[
                "viz_extraction_status"
            ],

        "extracted_field_count":
            record[
                "extracted_field_count"
            ],

        "viz_selected_variant":
            record.get(
                "viz_selected_variant"
            ),

        "viz_fallback_used":
            record.get(
                "viz_fallback_used"
            ),
    }

    for field_name in (
        "passport_number",
        "surname",
        "given_names",
        "nationality",
        "date_of_birth",
        "sex",
        "date_of_expiry",
    ):

        row[
            field_name
        ] = record.get(
            field_name
        )

        metadata = (
            record.get(
                f"{field_name}_metadata"
            )
            or {}
        )

        row[
            f"{field_name}_method"
        ] = metadata.get(
            "method"
        )

        row[
            f"{field_name}_variant"
        ] = metadata.get(
            "variant"
        )

        row[
            f"{field_name}_score"
        ] = metadata.get(
            "score"
        )

        row[
            f"{field_name}_variant_agreement"
        ] = metadata.get(
            "variant_agreement"
        )

        row[
            f"{field_name}_label_text"
        ] = metadata.get(
            "label_text"
        )

        row[
            f"{field_name}_raw_text"
        ] = metadata.get(
            "raw_text"
        )

        row[
            f"{field_name}_ocr_confidence"
        ] = metadata.get(
            "ocr_confidence"
        )

    return row


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
        flatten_for_csv(
            record
        )
        for record in records
    ]

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
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

    if not VIZ_JSON.exists():

        raise FileNotFoundError(
            f"Không thấy VIZ OCR JSON:\n"
            f"{VIZ_JSON}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with VIZ_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    if not isinstance(
        data,
        list,
    ):

        raise ValueError(
            "viz_ocr_full.json "
            "không phải JSON list."
        )

    records = []

    print(
        "=" * 76
    )

    print(
        "VIZ STRUCTURED FIELD EXTRACTION"
    )

    print(
        "=" * 76
    )

    print(
        f"Input records : "
        f"{len(data)}"
    )

    print()

    for (
        index,
        record,
    ) in enumerate(
        data,
        start=1,
    ):

        try:

            output = (
                process_record(
                    record
                )
            )

        except Exception as error:

            output = {
                "filename":
                    record.get(
                        "filename"
                    ),

                "viz_extraction_status":
                    "error",

                "extracted_field_count":
                    0,

                "error":
                    repr(
                        error
                    ),
            }

        records.append(
            output
        )

        print(
            f"["
            f"{index:>4}"
            f"/"
            f"{len(data)}"
            f"] "
            f"{output.get('filename')} "
            f"-> "
            f"{output.get('viz_extraction_status')} "
            f"| fields="
            f"{output.get('extracted_field_count')}"
        )

    write_json(
        records
    )

    write_csv(
        records
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    status_counts = {}

    for record in records:

        status = str(
            record.get(
                "viz_extraction_status"
            )
        )

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

    field_names = (
        "passport_number",
        "surname",
        "given_names",
        "nationality",
        "date_of_birth",
        "sex",
        "date_of_expiry",
    )

    field_counts = {
        field_name: 0
        for field_name
        in field_names
    }

    for record in records:

        for field_name in (
            field_names
        ):

            if record.get(
                field_name
            ):

                field_counts[
                    field_name
                ] += 1

    print()
    print(
        "=" * 76
    )

    print(
        "VIZ FIELD EXTRACTION SUMMARY"
    )

    print(
        "=" * 76
    )

    print()
    print(
        "STATUS"
    )

    print(
        "-" * 76
    )

    for (
        status,
        count,
    ) in sorted(
        status_counts.items()
    ):

        print(
            f"{status:<24}: "
            f"{count}"
        )

    print()
    print(
        "FIELD COVERAGE"
    )

    print(
        "-" * 76
    )

    denominator = max(
        1,
        len(records),
    )

    for field_name in (
        field_names
    ):

        count = (
            field_counts[
                field_name
            ]
        )

        percentage = (
            count
            / denominator
            * 100.0
        )

        print(
            f"{field_name:<24}: "
            f"{count:>4}"
            f"/"
            f"{len(records):<4} "
            f"("
            f"{percentage:6.2f}%"
            f")"
        )

    print()

    print(
        f"CSV : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"JSON: "
        f"{OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()