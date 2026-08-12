from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Any


MRZ_VALUE_FIELD = {
    "passport_number": "passport_number",
    "surname": "surname",
    "given_names": "given_names",
    "nationality": "nationality",
    "date_of_birth": "birth_date",
    "sex": "sex",
    "date_of_expiry": "expiry_date",
}

SOURCE_COMPARE_FIELDS = tuple(MRZ_VALUE_FIELD)

QUALITY_WEIGHTS = {
    "verified": 1.00,
    "strong": 0.92,
    "high": 0.88,
    "medium": 0.72,
    "weak": 0.48,
    "invalid": 0.20,
    "missing": 0.00,
    "high_confidence": 0.90,
    "medium_confidence": 0.72,
    "low_confidence": 0.45,
    "no_date_found": 0.00,
    "no_ocr_items": 0.00,
}

SEVERITY_PENALTY = {
    "high": 30.0,
    "medium": 12.0,
    "low": 5.0,
}


def _empty_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "null", "nan"}:
            return None
        return stripped
    return value


def _parse_bool(value: Any) -> bool | None:
    normalized = _empty_to_none(value)
    if normalized is None:
        return None
    text = str(normalized).lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_for_compare(field_name: str, value: Any) -> str | None:
    value = _empty_to_none(value)
    if value is None:
        return None

    text = _strip_accents(str(value)).upper().strip()

    if field_name == "passport_number":
        return re.sub(r"[^A-Z0-9]", "", text) or None

    if field_name in {"surname", "given_names", "nationality"}:
        text = text.replace("<", " ")
        text = re.sub(r"[^A-Z0-9]+", " ", text)
        return " ".join(text.split()) or None

    if field_name == "sex":
        return text[:1] if text else None

    if field_name in {"date_of_birth", "date_of_expiry", "date_of_issue"}:
        return text

    return " ".join(text.split()) or None


def parse_iso_date(value: Any) -> date | None:
    value = _empty_to_none(value)
    if value is None:
        return None

    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _mrz_source_strength(field_name: str, mrz_row: dict[str, Any]) -> str:
    if field_name == "passport_number":
        check = _parse_bool(mrz_row.get("passport_number_check_valid"))
    elif field_name == "date_of_birth":
        check = _parse_bool(mrz_row.get("birth_date_check_valid"))
    elif field_name == "date_of_expiry":
        check = _parse_bool(mrz_row.get("expiry_date_check_valid"))
    else:
        check = None

    parse_mode = _empty_to_none(mrz_row.get("parse_mode"))
    all_main = _parse_bool(mrz_row.get("all_main_checks_valid"))

    if check is True:
        return "verified"
    if check is False:
        return "invalid"

    if parse_mode == "strict_44_44" and all_main is True:
        return "verified"
    if parse_mode == "strict_44_44":
        return "strong"
    return "weak"


def _viz_source_strength(field_name: str, viz_row: dict[str, Any]) -> str:
    value = _empty_to_none(viz_row.get(field_name))
    if value is None:
        return "missing"

    try:
        score = float(viz_row.get(f"{field_name}_score"))
    except (TypeError, ValueError):
        score = None

    try:
        agreement = int(float(viz_row.get(f"{field_name}_variant_agreement")))
    except (TypeError, ValueError):
        agreement = 0

    if agreement >= 2 or (score is not None and score >= 8.0):
        return "high"
    if score is not None and score >= 6.0:
        return "medium"
    return "weak"


def _issue(
    code: str,
    severity: str,
    message: str,
    field: str | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "field": field,
        "message": message,
    }


def analyze_source_conflicts(
    mrz_row: dict[str, Any],
    viz_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[dict[str, Any]] = []
    conflict_fields: list[str] = []

    for field_name in SOURCE_COMPARE_FIELDS:
        mrz_value = normalize_for_compare(
            field_name,
            mrz_row.get(MRZ_VALUE_FIELD[field_name]),
        )
        viz_value = normalize_for_compare(field_name, viz_row.get(field_name))

        if mrz_value is None or viz_value is None or mrz_value == viz_value:
            continue

        mrz_strength = _mrz_source_strength(field_name, mrz_row)
        viz_strength = _viz_source_strength(field_name, viz_row)

        if mrz_strength == "verified" and viz_strength == "high":
            severity = "high"
        elif mrz_strength in {"verified", "strong"} and viz_strength in {
            "high",
            "medium",
        }:
            severity = "medium"
        else:
            severity = "low"

        conflict_fields.append(field_name)
        issues.append(
            _issue(
                code=f"source_conflict_{field_name}",
                severity=severity,
                field=field_name,
                message=(
                    f"MRZ={mrz_value!r} khác VIZ={viz_value!r} "
                    f"(mrz={mrz_strength}, viz={viz_strength})."
                ),
            )
        )

    return issues, conflict_fields


def analyze_temporal_consistency(
    final_fields: dict[str, Any],
    reference_date: date | None = None,
) -> list[dict[str, Any]]:
    if reference_date is None:
        reference_date = date.today()

    issues: list[dict[str, Any]] = []
    birth = parse_iso_date(final_fields.get("date_of_birth"))
    issue_date = parse_iso_date(final_fields.get("date_of_issue"))
    expiry = parse_iso_date(final_fields.get("date_of_expiry"))

    if final_fields.get("date_of_birth") and birth is None:
        issues.append(
            _issue(
                "invalid_date_of_birth_format",
                "medium",
                "date_of_birth không phải ISO date hợp lệ.",
                "date_of_birth",
            )
        )

    if final_fields.get("date_of_issue") and issue_date is None:
        issues.append(
            _issue(
                "invalid_date_of_issue_format",
                "medium",
                "date_of_issue không phải ISO date hợp lệ.",
                "date_of_issue",
            )
        )

    if final_fields.get("date_of_expiry") and expiry is None:
        issues.append(
            _issue(
                "invalid_date_of_expiry_format",
                "medium",
                "date_of_expiry không phải ISO date hợp lệ.",
                "date_of_expiry",
            )
        )

    if birth and birth > reference_date:
        issues.append(
            _issue(
                "birth_date_in_future",
                "high",
                "Ngày sinh nằm trong tương lai.",
                "date_of_birth",
            )
        )

    if issue_date and issue_date > reference_date:
        issues.append(
            _issue(
                "issue_date_in_future",
                "medium",
                "Ngày cấp nằm trong tương lai so với ngày chạy pipeline.",
                "date_of_issue",
            )
        )

    if birth and issue_date and issue_date < birth:
        issues.append(
            _issue(
                "issue_before_birth",
                "high",
                "Ngày cấp hộ chiếu sớm hơn ngày sinh.",
            )
        )

    if birth and expiry and expiry <= birth:
        issues.append(
            _issue(
                "expiry_not_after_birth",
                "high",
                "Ngày hết hạn không nằm sau ngày sinh.",
            )
        )

    if issue_date and expiry:
        if expiry <= issue_date:
            issues.append(
                _issue(
                    "expiry_not_after_issue",
                    "high",
                    "Ngày hết hạn không nằm sau ngày cấp.",
                )
            )
        else:
            validity_years = (expiry - issue_date).days / 365.25
            if validity_years < 0.25 or validity_years > 15.0:
                issues.append(
                    _issue(
                        "unusual_passport_validity_period",
                        "medium",
                        f"Thời hạn issue→expiry bất thường: {validity_years:.2f} năm.",
                    )
                )

    if birth and issue_date and issue_date >= birth:
        age_at_issue = (issue_date - birth).days / 365.25
        if age_at_issue > 120:
            issues.append(
                _issue(
                    "implausible_age_at_issue",
                    "high",
                    f"Tuổi tại ngày cấp bất thường: {age_at_issue:.1f}.",
                )
            )

    return issues


def analyze_document_consistency(
    final_fields: dict[str, Any],
    mrz_row: dict[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    document_type = normalize_for_compare(
        "document_type",
        mrz_row.get("document_type"),
    )
    if document_type and not document_type.startswith("P"):
        issues.append(
            _issue(
                "non_passport_mrz_document_type",
                "medium",
                f"MRZ document_type={document_type!r} không bắt đầu bằng P.",
            )
        )

    sex = normalize_for_compare("sex", final_fields.get("sex"))
    if sex is not None and sex not in {"M", "F", "X"}:
        issues.append(
            _issue(
                "unexpected_sex_value",
                "medium",
                f"Sex={sex!r} không thuộc M/F/X.",
                "sex",
            )
        )

    return issues


def compute_quality_score(
    field_quality: dict[str, str],
    issues: list[dict[str, Any]],
) -> tuple[float, str]:
    extracted_qualities = [
        QUALITY_WEIGHTS.get(str(quality), 0.45)
        for quality in field_quality.values()
        if str(quality) != "missing"
    ]

    if not extracted_qualities:
        return 0.0, "low_confidence"

    score = 100.0 * sum(extracted_qualities) / len(extracted_qualities)

    for issue in issues:
        score -= SEVERITY_PENALTY.get(str(issue.get("severity")), 5.0)

    score = max(0.0, min(100.0, score))
    has_high = any(issue.get("severity") == "high" for issue in issues)

    if score >= 90 and not has_high:
        status = "high_confidence"
    elif score >= 75 and not has_high:
        status = "medium_confidence"
    elif score >= 50:
        status = "review"
    else:
        status = "low_confidence"

    return round(score, 2), status


def analyze_final_record(
    final_fields: dict[str, Any],
    field_quality: dict[str, str],
    mrz_row: dict[str, Any],
    viz_row: dict[str, Any],
) -> dict[str, Any]:
    source_issues, conflict_fields = analyze_source_conflicts(mrz_row, viz_row)
    temporal_issues = analyze_temporal_consistency(final_fields)
    document_issues = analyze_document_consistency(final_fields, mrz_row)

    issues = source_issues + temporal_issues + document_issues
    quality_score, quality_status = compute_quality_score(field_quality, issues)

    severity_counts = {
        severity: sum(issue.get("severity") == severity for issue in issues)
        for severity in ("high", "medium", "low")
    }

    return {
        "consistency_issues": issues,
        "source_conflict_fields": conflict_fields,
        "consistency_issue_count": len(issues),
        "consistency_high_count": severity_counts["high"],
        "consistency_medium_count": severity_counts["medium"],
        "consistency_low_count": severity_counts["low"],
        "quality_score": quality_score,
        "quality_status": quality_status,
    }
