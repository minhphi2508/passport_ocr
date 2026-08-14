from __future__ import annotations

import csv
import json
import re
import unicodedata
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIZ_JSON = PROJECT_ROOT / "outputs" / "viz_ocr" / "viz_ocr_full.json"
VIZ_JSONL = PROJECT_ROOT / "outputs" / "viz_ocr" / "viz_ocr_records.jsonl"

MRZ_PARSED_CSV = (
    PROJECT_ROOT / "outputs" / "mrz_parsed" / "mrz_parsed_results.csv"
)
MRZ_VALIDATED_CSV = (
    PROJECT_ROOT / "outputs" / "mrz_validated" / "mrz_validated_results.csv"
)
VIZ_FIELDS_CSV = (
    PROJECT_ROOT / "outputs" / "viz_fields" / "viz_fields_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "date_of_issue_hybrid_v3"
)
OUTPUT_CSV = OUTPUT_DIR / "date_of_issue_hybrid_v3_results.csv"
OUTPUT_JSON = OUTPUT_DIR / "date_of_issue_hybrid_v3_results.json"

# Keep the historical output folder/file names so the rest of the pipeline
# remains backward compatible. Internally this is DOI extractor V5.
EXTRACTOR_VERSION = "doi_v6_1"

ISSUE_LABELS = (
    "date of issue",
    "issue date",
    "date issued",
    "date of issuance",
    "issuing date",
    "iss date",
    "iss. date",
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
    "date of grant",
    "grant date",
)

NEGATIVE_DATE_LABELS = (
    "date of birth",
    "birth date",
    "date de naissance",
    "fecha de nacimiento",
    "datum urodzenia",
    "geburtsdatum",
    "date of expiry",
    "expiry date",
    "expiration date",
    "date of expiration",
    "date d expiration",
    "fecha de expiracion",
    "fecha de caducidad",
    "valid until",
    "valid thru",
    "valid through",
)

MONTHS = {
    "jan": 1, "january": 1, "janvier": 1, "januar": 1,
    "feb": 2, "february": 2, "fevrier": 2, "februar": 2,
    "mar": 3, "march": 3, "mars": 3, "marzo": 3,
    "apr": 4, "april": 4, "avr": 4, "avril": 4, "abril": 4,
    "may": 5, "mai": 5, "mayo": 5,
    "jun": 6, "june": 6, "juin": 6, "junio": 6, "juni": 6,
    "jul": 7, "july": 7, "juillet": 7, "julio": 7, "juli": 7,
    "aug": 8, "august": 8, "aout": 8, "agosto": 8,
    "sep": 9, "sept": 9, "september": 9, "septembre": 9,
    "oct": 10, "october": 10, "octobre": 10, "octubre": 10, "okt": 10,
    "nov": 11, "november": 11, "novembre": 11, "noviembre": 11,
    "dec": 12, "december": 12, "decembre": 12, "diciembre": 12, "dez": 12,
}

VARIANT_ORDER = ("enhanced", "color", "grayscale")

OCR_MONTH_EQUIVALENTS = {
    "0ct": "oct",
    "oet": "oct",
    "n0v": "nov",
    "noy": "nov",
    "nar": "mar",
}

def resolve_month_token(token: str) -> int | None:
    token = normalize_text(token).replace(" ", "")
    if not token:
        return None
    token = OCR_MONTH_EQUIVALENTS.get(token, token)
    if token in MONTHS:
        return MONTHS[token]

    # OCR often emits bilingual month strings such as ME/JUN where one side
    # is damaged but the other side is reliable.
    for part in token.split("/"):
        part = OCR_MONTH_EQUIVALENTS.get(part, part)
        if part in MONTHS:
            return MONTHS[part]

    # Conservative fuzzy rescue for short month tokens only.
    best_month = None
    best_score = 0.0
    for alias, month in MONTHS.items():
        if len(alias) < 3:
            continue
        score = SequenceMatcher(None, token, alias).ratio()
        if score > best_score:
            best_score = score
            best_month = month
    if best_score >= 0.72:
        return best_month
    return None



def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9/.\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


NORMALIZED_ISSUE_LABELS = tuple(normalize_text(x) for x in ISSUE_LABELS)
NORMALIZED_NEGATIVE_LABELS = tuple(normalize_text(x) for x in NEGATIVE_DATE_LABELS)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _safe_iso(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _safe_date(year: int, month: int, day: int) -> str | None:
    # Reject OCR garbage such as 6206-11-06.
    current = date.today().year
    if year < 1900 or year > current + 1:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _normalize_two_digit_year(
    yy: int,
    birth_iso: str | None,
    expiry_iso: str | None,
) -> int:
    options = [1900 + yy, 2000 + yy]
    valid: list[int] = []
    birth = date.fromisoformat(birth_iso) if birth_iso else None
    expiry = date.fromisoformat(expiry_iso) if expiry_iso else None

    for year in options:
        if birth and year < birth.year:
            continue
        if expiry and year > expiry.year:
            continue
        if year > date.today().year + 1:
            continue
        valid.append(year)

    if valid:
        return max(valid)

    return min(options, key=lambda y: abs(y - date.today().year))


def _parse_numeric(
    day: int,
    month: int,
    year_text: str,
    birth_iso: str | None,
    expiry_iso: str | None,
) -> str | None:
    if len(year_text) not in {2, 4}:
        return None
    year = int(year_text)
    if len(year_text) == 2:
        year = _normalize_two_digit_year(year, birth_iso, expiry_iso)
    return _safe_date(year, month, day)


def extract_dates_from_text(
    text: str,
    birth_iso: str | None = None,
    expiry_iso: str | None = None,
) -> list[dict[str, str]]:
    normalized = normalize_text(text)
    found: dict[str, dict[str, str]] = {}

    def add(raw: str, iso: str | None, start: int = -1, end: int = -1) -> None:
        if iso:
            found[iso] = {
                "raw": raw,
                "iso": iso,
                "start": str(start),
                "end": str(end),
            }

    for m in re.finditer(
        r"(?<!\d)(\d{1,2})[./\-](\d{1,2})[./\-](\d{2}|\d{4})(?!\d)",
        normalized,
    ):
        add(
            m.group(0),
            _parse_numeric(int(m.group(1)), int(m.group(2)), m.group(3),
                           birth_iso, expiry_iso),
            m.start(), m.end(),
        )

    for m in re.finditer(
        r"(?<!\d)(\d{1,2})\s+(\d{1,2})\s+(\d{2}|\d{4})(?!\d)",
        normalized,
    ):
        add(
            m.group(0),
            _parse_numeric(int(m.group(1)), int(m.group(2)), m.group(3),
                           birth_iso, expiry_iso),
            m.start(), m.end(),
        )

    # OCR-glued numeric + textual month forms, e.g.
    #   127/JUL 2024 -> "12 7/JUL 2024" -> 2024-07-12
    #
    # The first two digits are treated as DD when valid. The remaining
    # 1-2 digits are treated as a duplicated numeric month marker and are
    # checked against the textual month when possible. This prevents the
    # generic textual-month regex from incorrectly reading the same token
    # as day=1.
    for m in re.finditer(
        r"(?<!\d)(\d{3,4})\s*/\s*([a-z0-9]{2,10})\s+(\d{2}|\d{4})(?!\d)",
        normalized,
    ):
        prefix = m.group(1)
        textual_month = resolve_month_token(m.group(2))

        if textual_month and len(prefix) >= 3:
            try:
                day = int(prefix[:2])
                numeric_month = int(prefix[2:])
            except ValueError:
                day = 0
                numeric_month = -1

            if (
                1 <= day <= 31
                and 1 <= numeric_month <= 12
                and numeric_month == textual_month
            ):
                add(
                    m.group(0),
                    _parse_numeric(
                        day,
                        textual_month,
                        m.group(3),
                        birth_iso,
                        expiry_iso,
                    ),
                    m.start(),
                    m.end(),
                )

    # OCR-tolerant textual month. Accepts 0CT, NAR and bilingual ME/JUN.
    for m in re.finditer(
        r"(?<![a-z0-9])(\d{1,2})\s*([a-z0-9]{2,10}(?:\s*/\s*[a-z0-9]{2,10})?)"
        r"\s*(\d{2}|\d{4})(?!\d)",
        normalized,
    ):
        month = resolve_month_token(m.group(2))
        if month:
            add(
                m.group(0),
                _parse_numeric(int(m.group(1)), month, m.group(3),
                               birth_iso, expiry_iso),
                m.start(), m.end(),
            )

    # Compact DDMMYY / DDMMYYYY.
    for m in re.finditer(r"(?<!\d)(\d{6}|\d{8})(?!\d)", normalized):
        token = m.group(1)
        day = int(token[:2])
        month = int(token[2:4])
        year_text = token[4:]
        add(
            token,
            _parse_numeric(day, month, year_text, birth_iso, expiry_iso),
            m.start(), m.end(),
        )

    return list(found.values())

def label_similarity(text: str) -> float:
    normalized = normalize_text(text)
    if len(normalized) < 4:
        return 0.0
    best = 0.0
    for label in NORMALIZED_ISSUE_LABELS:
        if label in normalized or normalized in label:
            best = max(best, 1.0)
        else:
            best = max(best, SequenceMatcher(None, normalized, label).ratio())
    return best


def has_issue_signal(text: str) -> bool:
    normalized = normalize_text(text)
    compact = normalized.replace(" ", "")
    for label in NORMALIZED_ISSUE_LABELS:
        if label in normalized or label.replace(" ", "") in compact:
            return True

    if (
        "date" in normalized
        and any(k in normalized for k in ("issue", "issu", "iss "))
    ):
        return True

    # OCR typo rescue: DATE OF ISHUE, DUSE OF ISSUE, etc.
    best = max(
        (
            SequenceMatcher(None, normalized, label).ratio()
            for label in NORMALIZED_ISSUE_LABELS
        ),
        default=0.0,
    )
    return best >= 0.70

def has_negative_signal(text: str) -> bool:
    normalized = normalize_text(text)
    compact = normalized.replace(" ", "")
    return any(
        p in normalized or p.replace(" ", "") in compact
        for p in NORMALIZED_NEGATIVE_LABELS
    )


def _box(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = item.get("box")
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, value)
        return x1, y1, x2, y2
    except (TypeError, ValueError):
        return None


def _spatial_score(
    label_item: dict[str, Any],
    date_item: dict[str, Any],
) -> float | None:
    lb = _box(label_item)
    db = _box(date_item)
    if lb is None or db is None:
        return None

    lx1, ly1, lx2, ly2 = lb
    dx1, dy1, dx2, dy2 = db
    lh = max(1.0, ly2 - ly1)
    lw = max(1.0, lx2 - lx1)
    dh = max(1.0, dy2 - dy1)

    # Same row, value to the right.
    vertical_delta = abs(((dy1 + dy2) / 2) - ((ly1 + ly2) / 2))
    horizontal_gap = dx1 - lx2
    if (
        -0.5 * lh <= horizontal_gap <= 5.0 * lh
        and vertical_delta <= 1.25 * max(lh, dh)
    ):
        return 34.0 - max(0.0, horizontal_gap) / max(lh, 1.0) * 3.0

    # Value immediately below the label in roughly the same column.
    gap = dy1 - ly2
    overlap = max(0.0, min(lx2, dx2) - max(lx1, dx1))
    overlap_ratio = overlap / max(1.0, min(lw, dx2 - dx1))
    center_delta = abs(((dx1 + dx2) / 2) - ((lx1 + lx2) / 2))
    if (
        -0.35 * lh <= gap <= 4.5 * lh
        and (overlap_ratio >= 0.10 or center_delta <= max(lw, 180.0))
    ):
        return (
            31.0
            - max(0.0, gap) / lh * 4.0
            + min(10.0, overlap_ratio * 10.0)
        )

    return None


def _temporal_allowed(
    candidate_iso: str,
    birth_iso: str | None,
    expiry_iso: str | None,
) -> bool:
    candidate = date.fromisoformat(candidate_iso)
    if candidate > date.today() + timedelta(days=366):
        return False
    if birth_iso and candidate <= date.fromisoformat(birth_iso):
        return False
    if expiry_iso and candidate >= date.fromisoformat(expiry_iso):
        return False
    return True


def _validity_bonus(
    candidate_iso: str,
    expiry_iso: str | None,
) -> float:
    if not expiry_iso:
        return 0.0
    candidate = date.fromisoformat(candidate_iso)
    expiry = date.fromisoformat(expiry_iso)
    years = (expiry - candidate).days / 365.2425

    bonus = 0.0
    # Passport validity is commonly around 5 or 10 years; do not make this
    # a hard rule because some countries/ages use other periods.
    for target in (5.0, 10.0):
        distance = abs(years - target)
        if distance <= 0.03:       # ~11 days
            bonus = max(bonus, 28.0)
        elif distance <= 0.15:     # ~55 days
            bonus = max(bonus, 20.0)
        elif distance <= 0.50:
            bonus = max(bonus, 8.0)

    # Conservative tie-breaker only when the candidate is already close to
    # a typical 5/10-year validity interval. Exact day/month alignment with
    # the checksum-backed expiry is strong supporting evidence.
    if (
        candidate.day == expiry.day
        and candidate.month == expiry.month
        and any(abs(years - target) <= 0.15 for target in (5.0, 10.0))
    ):
        bonus += 10.0

    return bonus


def _candidate_is_known_non_issue(
    iso: str,
    birth_iso: str | None,
    expiry_iso: str | None,
) -> bool:
    return iso == birth_iso or iso == expiry_iso


def _read_csv_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {
        str(r.get("filename") or ""): r
        for r in rows
        if r.get("filename")
    }


def _filename_keys(filename: str | None) -> list[str]:
    text = str(filename or "")
    if not text:
        return []
    stem = Path(text).stem
    keys = [text, stem, f"{stem}.jpg", f"{stem}.png"]
    return list(dict.fromkeys(keys))


def _lookup(
    mapping: dict[str, dict[str, str]],
    filename: str | None,
) -> dict[str, str]:
    for key in _filename_keys(filename):
        if key in mapping:
            return mapping[key]
    return {}


def choose_temporal_context(
    filename: str | None,
    mrz_parsed: dict[str, dict[str, str]],
    mrz_validated: dict[str, dict[str, str]],
    viz_fields: dict[str, dict[str, str]],
) -> tuple[str | None, str | None, str]:
    parsed = _lookup(mrz_parsed, filename)
    validated = _lookup(mrz_validated, filename)
    viz = _lookup(viz_fields, filename)

    mrz_birth = _safe_iso(parsed.get("birth_date"))
    mrz_expiry = _safe_iso(parsed.get("expiry_date"))
    viz_birth = _safe_iso(viz.get("date_of_birth"))
    viz_expiry = _safe_iso(viz.get("date_of_expiry"))

    all_checks = _as_bool(validated.get("all_main_checks_valid"))
    strict = str(parsed.get("parse_mode") or "") == "strict_44_44"

    if all_checks is True and strict:
        birth = mrz_birth or viz_birth
        expiry = mrz_expiry or viz_expiry
        if birth and expiry and date.fromisoformat(birth) >= date.fromisoformat(expiry):
            birth = None
        return birth, expiry, "validated_mrz"

    birth = viz_birth or mrz_birth
    expiry = viz_expiry or mrz_expiry
    source_parts = []
    if birth:
        source_parts.append("birth_viz" if viz_birth else "birth_mrz")
    if expiry:
        source_parts.append("expiry_viz" if viz_expiry else "expiry_mrz")

    # Never let an internally impossible context reject otherwise plausible DOI.
    if birth and expiry and date.fromisoformat(birth) >= date.fromisoformat(expiry):
        birth = None
        source_parts.append("birth_discarded_invalid")

    return birth, expiry, "+".join(source_parts) or "none"

def _variant_records(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    variants = record.get("variants") or {}
    result: list[tuple[str, dict[str, Any]]] = []

    if isinstance(variants, dict):
        for name in VARIANT_ORDER:
            value = variants.get(name)
            if isinstance(value, dict):
                result.append((name, value))
        for name, value in variants.items():
            if name not in VARIANT_ORDER and isinstance(value, dict):
                result.append((str(name), value))

    if not result and isinstance(record.get("selected_result"), dict):
        result.append((
            str(record.get("selected_variant") or "selected"),
            record["selected_result"],
        ))

    return result



def _union_box(items: list[dict[str, Any]]) -> list[float] | None:
    boxes = [_box(item) for item in items]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _date_item_windows(
    items: list[dict[str, Any]],
    birth_iso: str | None,
    expiry_iso: str | None,
) -> list[tuple[int, dict[str, Any], dict[str, str]]]:
    out = []
    seen = set()
    for start in range(len(items)):
        for size in (1, 2, 3):
            chunk = items[start:start + size]
            if len(chunk) != size or not all(isinstance(x, dict) for x in chunk):
                continue
            text = " ".join(str(x.get("text") or "") for x in chunk).strip()
            if not text:
                continue
            synthetic = {
                "text": text,
                "box": _union_box(chunk),
                "confidence": min(
                    [float(x.get("confidence") or 0.0) for x in chunk] or [0.0]
                ),
            }
            for parsed in extract_dates_from_text(text, birth_iso, expiry_iso):
                key = (parsed["iso"], start, size)
                if key in seen:
                    continue
                seen.add(key)
                out.append((start, synthetic, parsed))
    return out


def _all_parsed_dates(
    record: dict[str, Any],
    birth_iso: str | None,
    expiry_iso: str | None,
) -> list[dict[str, Any]]:
    values = []
    for variant_name, variant in _variant_records(record):
        items = variant.get("items") or []
        if not isinstance(items, list):
            continue
        for idx, item, parsed in _date_item_windows(items, birth_iso, expiry_iso):
            values.append({
                "iso": parsed["iso"],
                "raw": parsed["raw"],
                "variant": variant_name,
                "item_index": idx,
                "candidate_text": item.get("text"),
            })
    return values


def structural_role_candidates(
    record: dict[str, Any],
    birth_iso: str | None,
    expiry_iso: str | None,
    context_source: str,
) -> list[dict[str, Any]]:
    parsed = _all_parsed_dates(record, birth_iso, expiry_iso)
    unique = {}
    for item in parsed:
        iso = item["iso"]
        if iso == birth_iso or iso == expiry_iso:
            continue
        if not _temporal_allowed(iso, birth_iso, expiry_iso):
            continue
        unique.setdefault(iso, item)

    if not unique:
        return []

    # If checksum-valid MRZ gives DOB/expiry and OCR contains exactly one other
    # date between them, that remaining role is strongly indicative of DOI.
    if context_source == "validated_mrz" and len(unique) == 1:
        item = next(iter(unique.values()))
        return [{
            "iso": item["iso"],
            "raw": item["raw"],
            "score": 86.0,
            "method": "date_role_elimination",
            "label_text": None,
            "candidate_text": item.get("candidate_text"),
            "variant": item.get("variant"),
        }]

    return []


def collect_candidates(
    record: dict[str, Any],
    birth_iso: str | None,
    expiry_iso: str | None,
) -> tuple[list[dict[str, Any]], bool]:
    candidates: list[dict[str, Any]] = []
    any_issue_label = False

    for variant_name, variant in _variant_records(record):
        items = variant.get("items") or []
        if not isinstance(items, list):
            continue

        labels: list[tuple[int, dict[str, Any], float]] = []
        date_items: list[tuple[int, dict[str, Any], dict[str, str]]] = []

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "")
            if has_negative_signal(text):
                pass
            elif has_issue_signal(text):
                similarity = label_similarity(text)
                labels.append((idx, item, similarity))
                any_issue_label = True

        date_items = _date_item_windows(
            items,
            birth_iso,
            expiry_iso,
        )

        for date_idx, date_item, parsed in date_items:
            iso = parsed["iso"]
            if not _temporal_allowed(iso, birth_iso, expiry_iso):
                continue
            if _candidate_is_known_non_issue(iso, birth_iso, expiry_iso):
                continue

            best_score = -1e9
            best_label_text = None
            best_method = None

            for label_idx, label_item, similarity in labels:
                label_text = str(label_item.get("text") or "")

                # If OCR merged a previous date and the "Date of issue" label
                # into one item, a date occurring before the label is usually
                # DOB/another field, not DOI.
                if label_idx == date_idx:
                    normalized = normalize_text(label_text)
                    issue_positions = [
                        normalized.find(lbl)
                        for lbl in NORMALIZED_ISSUE_LABELS
                        if lbl in normalized
                    ]
                    issue_pos = min(issue_positions) if issue_positions else -1
                    try:
                        date_pos = int(parsed.get("start", "-1"))
                    except ValueError:
                        date_pos = -1
                    if (
                        issue_pos >= 0
                        and date_pos >= 0
                        and date_pos + len(parsed["raw"]) <= issue_pos
                    ):
                        continue

                spatial = _spatial_score(label_item, date_item)
                index_distance = abs(label_idx - date_idx)
                score = similarity * 45.0
                method = "issue_label_nearby"

                if spatial is not None:
                    score += spatial
                    method = "issue_label_spatial"
                else:
                    # Order-only fallback is deliberately conservative.
                    if index_distance > 3:
                        continue
                    score += max(0.0, 18.0 - index_distance * 5.0)

                score += _validity_bonus(iso, expiry_iso)

                try:
                    score += float(date_item.get("confidence") or 0.0) * 5.0
                except (TypeError, ValueError):
                    pass

                if score > best_score:
                    best_score = score
                    best_label_text = label_text
                    best_method = method

            # If no issue label is available, temporal-only rescue is allowed
            # only when expiry relationship is exceptionally strong.
            if best_method is None:
                validity = _validity_bonus(iso, expiry_iso)
                if validity < 28.0:
                    continue
                best_score = 42.0 + validity
                best_method = "temporal_validity_rescue"

            candidates.append({
                "iso": iso,
                "raw": parsed["raw"],
                "score": float(best_score),
                "method": best_method,
                "label_text": best_label_text,
                "candidate_text": str(date_item.get("text") or ""),
                "variant": variant_name,
            })

    return candidates, any_issue_label


def _replace_year_safe(value: date, year: int) -> date | None:
    try:
        return value.replace(year=year)
    except ValueError:
        # Feb 29 -> Feb 28.
        if value.month == 2 and value.day == 29:
            return value.replace(year=year, day=28)
        return None


def expiry_backoff_candidates(
    record: dict[str, Any],
    birth_iso: str | None,
    expiry_iso: str | None,
    any_issue_label: bool,
) -> list[dict[str, Any]]:
    if not expiry_iso:
        return []

    expiry = date.fromisoformat(expiry_iso)
    all_text = " ".join(
        str(item.get("text") or "")
        for _, variant in _variant_records(record)
        for item in (variant.get("items") or [])
        if isinstance(item, dict)
    )
    normalized_all = normalize_text(all_text)
    parsed = _all_parsed_dates(record, birth_iso, expiry_iso)

    derived: list[dict[str, Any]] = []

    for years in (5, 10):
        base = _replace_year_safe(expiry, expiry.year - years)
        if base is None:
            continue

        for candidate in (base, base + timedelta(days=1)):
            iso = candidate.isoformat()
            if not _temporal_allowed(iso, birth_iso, expiry_iso):
                continue

            year4 = str(candidate.year)
            year2 = year4[-2:]

            year_support = (
                year4 in normalized_all
                or re.search(rf"(?<!\d){re.escape(year2)}(?!\d)", normalized_all)
                is not None
            )

            # Strong day/month support can come from a misread year (e.g.
            # 22 12 2010 vs expected 22 12 2019).
            day_month_support = False
            exact_day_support = False
            one_digit_year_error = False
            for item in parsed:
                if item["iso"] in {birth_iso, expiry_iso}:
                    continue
                d = date.fromisoformat(item["iso"])
                if d.month == candidate.month and abs(d.day - candidate.day) <= 1:
                    day_month_support = True
                if d.month == candidate.month and d.day == candidate.day:
                    exact_day_support = True
                    if len(str(d.year)) == 4:
                        diffs = sum(a != b for a, b in zip(str(d.year), year4))
                        if diffs == 1:
                            one_digit_year_error = True

            # Even if the month token itself is corrupted (e.g. NEOUN), a
            # matching day+year around an alphabetic token is useful evidence.
            day_year_support = (
                re.search(
                    rf"(?<!\d){candidate.day:02d}\s+[a-z0-9/]+\s+{year4}(?!\d)",
                    normalized_all,
                )
                is not None
                or re.search(
                    rf"(?<!\d){candidate.day}\s+[a-z0-9/]+\s+{year2}(?!\d)",
                    normalized_all,
                )
                is not None
            )

            # Month-only evidence near a recognized issue label is enough for
            # a conservative 10-year rescue when OCR lost day/year entirely.
            month_names = [
                alias for alias, month in MONTHS.items()
                if month == candidate.month and len(alias) == 3
            ]
            month_support = any(
                re.search(rf"\b{re.escape(alias)}\b", normalized_all)
                for alias in month_names
            )

            if not (
                year_support
                or day_year_support
                or one_digit_year_error
                or (any_issue_label and (day_month_support or month_support))
            ):
                continue

            score = 72.0
            if year_support:
                score += 10.0
            if day_year_support:
                score += 12.0
            if one_digit_year_error:
                score += 14.0
            if day_month_support:
                score += 5.0
            if exact_day_support:
                score += 7.0
            if any_issue_label:
                score += 5.0
            # When OCR has lost the day/year entirely, 10-year validity is
            # a modest tie-breaker, not a hard rule.
            if years == 10 and not year_support and not exact_day_support:
                score += 3.0

            derived.append({
                "iso": iso,
                "raw": None,
                "score": score,
                "method": f"expiry_backoff_{years}y_v6",
                "label_text": "date of issue / temporal rescue",
                "candidate_text": None,
                "variant": None,
            })

    return derived

def rank_candidates(
    candidates: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    best_by_iso: dict[str, dict[str, Any]] = {}
    support_by_iso: dict[str, set[str]] = {}

    items = list(candidates)
    for c in items:
        variant = c.get("variant")
        if variant:
            support_by_iso.setdefault(c["iso"], set()).add(str(variant))

    for c in items:
        candidate = dict(c)
        support = support_by_iso.get(candidate["iso"], set())
        candidate["variant_agreement_count"] = len(support)
        candidate["supporting_variants"] = sorted(support)
        candidate["score"] = float(candidate["score"]) + 7.0 * max(
            0, len(support) - 1
        )
        current = best_by_iso.get(candidate["iso"])
        if current is None or candidate["score"] > current["score"]:
            best_by_iso[candidate["iso"]] = candidate

    return sorted(
        best_by_iso.values(),
        key=lambda x: float(x["score"]),
        reverse=True,
    )


def extract_one(
    record: dict[str, Any],
    mrz_parsed: dict[str, dict[str, str]],
    mrz_validated: dict[str, dict[str, str]],
    viz_fields: dict[str, dict[str, str]],
) -> dict[str, Any]:
    filename = str(record.get("filename") or "")
    birth_iso, expiry_iso, context_source = choose_temporal_context(
        filename,
        mrz_parsed,
        mrz_validated,
        viz_fields,
    )

    candidates, any_issue_label = collect_candidates(
        record,
        birth_iso,
        expiry_iso,
    )

    candidates.extend(
        structural_role_candidates(
            record,
            birth_iso,
            expiry_iso,
            context_source,
        )
    )

    candidates.extend(
        expiry_backoff_candidates(
            record,
            birth_iso,
            expiry_iso,
            any_issue_label,
        )
    )

    ranked = rank_candidates(candidates)

    if not ranked:
        has_items = any(
            bool((variant.get("items") or []))
            for _, variant in _variant_records(record)
        )
        return {
            "filename": filename,
            "status": "no_date_found" if has_items else "no_ocr_items",
            "date_of_issue": None,
            "method": None,
            "score": None,
            "score_margin": None,
            "birth_date_mrz": _safe_iso(
                _lookup(mrz_parsed, filename).get("birth_date")
            ),
            "expiry_date_mrz": _safe_iso(
                _lookup(mrz_parsed, filename).get("expiry_date")
            ),
            "birth_date_context": birth_iso,
            "expiry_date_context": expiry_iso,
            "temporal_context_source": context_source,
            "extractor_version": EXTRACTOR_VERSION,
            "all_candidate_dates": [],
        }

    best = ranked[0]
    second = ranked[1]["score"] if len(ranked) > 1 else None
    margin = best["score"] - second if second is not None else None

    score = float(best["score"])
    if score >= 88.0 and (margin is None or margin >= 4.0):
        status = "high_confidence"
    elif score >= 65.0:
        status = "medium_confidence"
    else:
        status = "low_confidence"

    return {
        "filename": filename,
        "status": status,
        "date_of_issue": best["iso"],
        "raw_date": best.get("raw"),
        "method": best.get("method"),
        "score": round(score, 3),
        "score_margin": round(float(margin), 3) if margin is not None else None,
        "birth_date_mrz": _safe_iso(
            _lookup(mrz_parsed, filename).get("birth_date")
        ),
        "expiry_date_mrz": _safe_iso(
            _lookup(mrz_parsed, filename).get("expiry_date")
        ),
        "birth_date_context": birth_iso,
        "expiry_date_context": expiry_iso,
        "temporal_context_source": context_source,
        "label_text": best.get("label_text"),
        "candidate_text": best.get("candidate_text"),
        "all_candidate_dates": [x["iso"] for x in ranked],
        "spatial_variant": best.get("variant"),
        "spatial_variant_agreement_count": best.get("variant_agreement_count"),
        "spatial_supporting_variants": best.get("supporting_variants"),
        "extractor_version": EXTRACTOR_VERSION,
    }


def load_viz_records() -> list[dict[str, Any]]:
    if VIZ_JSON.exists():
        with VIZ_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data

    # Compatibility with append-only checkpoint builds.
    if VIZ_JSONL.exists():
        records = []
        with VIZ_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    # A partially written final line is safe to ignore.
                    continue
                if isinstance(value, dict):
                    records.append(value)
        if records:
            return records

    raise FileNotFoundError(
        f"Không thấy VIZ OCR records:\n{VIZ_JSON}\nhoặc\n{VIZ_JSONL}"
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_outputs(results: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fieldnames: list[str] = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                key: _csv_value(row.get(key))
                for key in fieldnames
            })


def main() -> None:
    records = load_viz_records()
    mrz_parsed = _read_csv_map(MRZ_PARSED_CSV)
    mrz_validated = _read_csv_map(MRZ_VALIDATED_CSV)
    viz_fields = _read_csv_map(VIZ_FIELDS_CSV)

    results = [
        extract_one(
            record,
            mrz_parsed,
            mrz_validated,
            viz_fields,
        )
        for record in records
    ]

    write_outputs(results)

    total = len(results)
    found = sum(bool(r.get("date_of_issue")) for r in results)
    high = sum(r.get("status") == "high_confidence" for r in results)
    medium = sum(r.get("status") == "medium_confidence" for r in results)
    low = sum(r.get("status") == "low_confidence" for r in results)

    print("=" * 76)
    print("DATE OF ISSUE EXTRACTION — V6.1")
    print("=" * 76)
    print(f"Samples          : {total}")
    print(f"DOI found        : {found}/{total} ({found/total:.1%})" if total else "DOI found: 0")
    print(f"High confidence  : {high}")
    print(f"Medium confidence: {medium}")
    print(f"Low confidence   : {low}")
    print()
    print(f"CSV : {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
