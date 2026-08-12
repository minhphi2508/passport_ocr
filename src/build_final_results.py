from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from consistency_checks import analyze_final_record
from sample_manifest import load_manifest, sample_id_from_generated_filename


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGE_ROOT = PROJECT_ROOT / "outputs" / "passport_pages_safe"
MANIFEST_CSV = PAGE_ROOT / "input_manifest.csv"
PROCESSING_CSV = PAGE_ROOT / "processing_results.csv"

MRZ_VALIDATED_CSV = (
    PROJECT_ROOT / "outputs" / "mrz_validated" / "mrz_validated_results.csv"
)
VIZ_FIELDS_CSV = PROJECT_ROOT / "outputs" / "viz_fields" / "viz_fields_results.csv"
DOI_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "date_of_issue_hybrid_v3"
    / "date_of_issue_hybrid_v3_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_results"
OUTPUT_CSV = OUTPUT_DIR / "passport_extraction_results.csv"
OUTPUT_JSON = OUTPUT_DIR / "passport_extraction_results.json"

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

MRZ_CHECK_FIELD = {
    "passport_number": "passport_number_check_valid",
    "date_of_birth": "birth_date_check_valid",
    "date_of_expiry": "expiry_date_check_valid",
}

MRZ_VALUE_FIELD = {
    "passport_number": "passport_number",
    "surname": "surname",
    "given_names": "given_names",
    "nationality": "nationality",
    "date_of_birth": "birth_date",
    "sex": "sex",
    "date_of_expiry": "expiry_date",
}


def load_csv(path: Path, required: bool = True) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Không thấy file:\n{path}")
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def empty_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "nan", "null"}:
            return None
        return stripped
    return value


def parse_optional_bool(value: Any) -> bool | None:
    normalized = empty_to_none(value)
    if normalized is None:
        return None
    text = str(normalized).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def parse_optional_float(value: Any) -> float | None:
    normalized = empty_to_none(value)
    if normalized is None:
        return None
    try:
        return float(normalized)
    except (TypeError, ValueError):
        return None


def parse_optional_int(value: Any) -> int | None:
    normalized = empty_to_none(value)
    if normalized is None:
        return None
    try:
        return int(float(normalized))
    except (TypeError, ValueError):
        return None


def index_by_sample_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}

    for row in rows:
        sample_id = empty_to_none(row.get("sample_id"))
        if sample_id is None:
            sample_id = sample_id_from_generated_filename(row.get("filename"))

        if sample_id:
            output[str(sample_id)] = row

    return output


def mrz_field_quality(field_name: str, mrz_row: dict[str, str]) -> str:
    value = empty_to_none(mrz_row.get(MRZ_VALUE_FIELD[field_name]))
    if value is None:
        return "missing"

    parse_status = empty_to_none(mrz_row.get("parse_status"))
    parse_mode = empty_to_none(mrz_row.get("parse_mode"))

    if parse_status != "success":
        return "weak"

    check_field = MRZ_CHECK_FIELD.get(field_name)
    if check_field is not None:
        check = parse_optional_bool(mrz_row.get(check_field))
        if check is True:
            return "verified"
        if check is False:
            return "invalid"
        return "medium" if parse_mode == "strict_44_44" else "weak"

    # Name/nationality/sex do not have independent TD3 check digits. A strict
    # 44+44 MRZ is strong structural evidence, but should not be labelled as
    # independently "verified".
    if parse_mode == "strict_44_44":
        return "strong"

    return "weak"


def viz_field_quality(field_name: str, viz_row: dict[str, str]) -> str:
    value = empty_to_none(viz_row.get(field_name))
    if value is None:
        return "missing"

    score = parse_optional_float(viz_row.get(f"{field_name}_score"))
    agreement = parse_optional_int(viz_row.get(f"{field_name}_variant_agreement")) or 0

    if agreement >= 2:
        return "high"
    if score is not None and score >= 8.0:
        return "high"
    if score is not None and score >= 6.0:
        return "medium"
    return "weak"


def choose_field(
    field_name: str,
    mrz_row: dict[str, str],
    viz_row: dict[str, str],
) -> tuple[Any, str | None, str]:
    mrz_value = empty_to_none(mrz_row.get(MRZ_VALUE_FIELD[field_name]))
    viz_value = empty_to_none(viz_row.get(field_name))
    mrz_quality = mrz_field_quality(field_name, mrz_row)
    viz_quality = viz_field_quality(field_name, viz_row)

    if mrz_value is not None and mrz_quality == "verified":
        return mrz_value, "mrz_verified", mrz_quality

    if mrz_value is not None and mrz_quality == "invalid":
        if viz_value is not None:
            return viz_value, f"viz_over_mrz_invalid_{viz_quality}", viz_quality
        return mrz_value, "mrz_invalid_fallback", mrz_quality

    if mrz_value is not None and mrz_quality == "strong":
        return mrz_value, "mrz_strong", mrz_quality

    if mrz_value is not None and mrz_quality == "medium":
        if viz_value is not None and viz_quality == "high":
            return viz_value, "viz_high_over_mrz_medium", viz_quality
        return mrz_value, "mrz_medium", mrz_quality

    if viz_value is not None:
        return viz_value, f"viz_{viz_quality}", viz_quality

    if mrz_value is not None:
        return mrz_value, "mrz_weak_fallback", mrz_quality

    return None, None, "missing"


def date_of_issue_quality(doi_row: dict[str, str]) -> str:
    if empty_to_none(doi_row.get("date_of_issue")) is None:
        return "missing"

    status = str(empty_to_none(doi_row.get("status")) or "low_confidence")
    if status in {"high_confidence", "medium_confidence", "low_confidence"}:
        return status
    return "weak"


def build_final_record(
    manifest_row: dict[str, str],
    processing_row: dict[str, str] | None,
    mrz_row: dict[str, str] | None,
    viz_row: dict[str, str] | None,
    doi_row: dict[str, str] | None,
) -> dict[str, Any]:
    processing_row = processing_row or {}
    mrz_row = mrz_row or {}
    viz_row = viz_row or {}
    doi_row = doi_row or {}

    sample_id = manifest_row["sample_id"]
    field_sources: dict[str, str | None] = {}
    field_quality: dict[str, str] = {}
    final_fields: dict[str, Any] = {}

    for field_name in (
        "passport_number",
        "surname",
        "given_names",
        "nationality",
        "date_of_birth",
        "sex",
        "date_of_expiry",
    ):
        value, source, quality = choose_field(field_name, mrz_row, viz_row)
        final_fields[field_name] = value
        field_sources[field_name] = source
        field_quality[field_name] = quality

    date_of_issue = empty_to_none(doi_row.get("date_of_issue"))
    final_fields["date_of_issue"] = date_of_issue
    field_sources["date_of_issue"] = "viz_doi" if date_of_issue else None
    field_quality["date_of_issue"] = date_of_issue_quality(doi_row)

    missing_fields = [
        field_name for field_name in CORE_FIELDS if final_fields[field_name] is None
    ]
    extracted_field_count = len(CORE_FIELDS) - len(missing_fields)

    if extracted_field_count == len(CORE_FIELDS):
        coverage_status = "complete"
    elif extracted_field_count > 0:
        coverage_status = "partial"
    else:
        coverage_status = "failed"

    mrz_available = bool(mrz_row)
    viz_available = bool(viz_row)

    if mrz_available and viz_available:
        extraction_mode = "mrz_plus_viz"
    elif mrz_available:
        extraction_mode = "mrz_only"
    elif viz_available:
        extraction_mode = "viz_only"
    else:
        extraction_mode = "none"

    all_main_checks_valid = parse_optional_bool(mrz_row.get("all_main_checks_valid"))
    parse_status = empty_to_none(mrz_row.get("parse_status"))
    parse_mode = empty_to_none(mrz_row.get("parse_mode"))

    quality = analyze_final_record(
        final_fields=final_fields,
        field_quality=field_quality,
        mrz_row=mrz_row,
        viz_row=viz_row,
    )

    review_reasons: list[str] = []

    if coverage_status != "complete":
        review_reasons.append("incomplete_fields")

    if quality["quality_status"] in {"review", "low_confidence"}:
        review_reasons.append(f"quality_status:{quality['quality_status']}")

    if all_main_checks_valid is False:
        review_reasons.append("mrz_checksum_failed")

    if parse_mode == "padded_or_truncated":
        review_reasons.append("mrz_soft_parse")

    if any(
        source is not None and "invalid" in source
        for source in field_sources.values()
    ):
        review_reasons.append("invalid_mrz_field_used_as_fallback")

    for issue in quality["consistency_issues"]:
        if issue["severity"] in {"high", "medium"}:
            review_reasons.append(f"consistency:{issue['code']}")

    page_status = empty_to_none(processing_row.get("status"))
    if page_status in {
        "no_passport_evidence",
        "image_read_failed",
        "unexpected_error",
        "empty_crop",
    }:
        review_reasons.append(f"passport_stage:{page_status}")

    # Deduplicate while preserving order.
    review_reasons = list(dict.fromkeys(review_reasons))

    return {
        "sample_id": sample_id,
        "filename": manifest_row["source_filename"],
        "relative_path": manifest_row["relative_path"],
        "generated_filename": f"{sample_id}.jpg",
        "passport_stage_status": page_status,
        "passport_gate_status": empty_to_none(
            processing_row.get("passport_gate_status")
        ),
        # final_status remains as a compatibility alias for old consumers.
        "final_status": coverage_status,
        "coverage_status": coverage_status,
        "quality_status": quality["quality_status"],
        "quality_score": quality["quality_score"],
        "extraction_mode": extraction_mode,
        "extracted_field_count": extracted_field_count,
        "missing_fields": missing_fields,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        **final_fields,
        "field_sources": field_sources,
        "field_quality": field_quality,
        "consistency_issues": quality["consistency_issues"],
        "source_conflict_fields": quality["source_conflict_fields"],
        "consistency_issue_count": quality["consistency_issue_count"],
        "consistency_high_count": quality["consistency_high_count"],
        "consistency_medium_count": quality["consistency_medium_count"],
        "consistency_low_count": quality["consistency_low_count"],
        "document_type": empty_to_none(mrz_row.get("document_type")),
        "issuing_country": empty_to_none(mrz_row.get("issuing_country")),
        "personal_number": empty_to_none(mrz_row.get("personal_number")),
        "mrz_available": mrz_available,
        "mrz_parse_status": parse_status,
        "mrz_parse_mode": parse_mode,
        "mrz_ocr_variant": empty_to_none(mrz_row.get("ocr_selected_variant")),
        "mrz_ocr_confidence": parse_optional_float(
            mrz_row.get("ocr_mean_confidence")
        ),
        "mrz_validation_status": empty_to_none(mrz_row.get("validation_status")),
        "passport_number_check_valid": parse_optional_bool(
            mrz_row.get("passport_number_check_valid")
        ),
        "birth_date_check_valid": parse_optional_bool(
            mrz_row.get("birth_date_check_valid")
        ),
        "expiry_date_check_valid": parse_optional_bool(
            mrz_row.get("expiry_date_check_valid")
        ),
        "personal_number_check_valid": parse_optional_bool(
            mrz_row.get("personal_number_check_valid")
        ),
        "final_check_valid": parse_optional_bool(mrz_row.get("final_check_valid")),
        "all_main_checks_valid": all_main_checks_valid,
        "mrz_line_1": empty_to_none(mrz_row.get("ocr_line_1")),
        "mrz_line_2": empty_to_none(mrz_row.get("ocr_line_2")),
        "viz_available": viz_available,
        "viz_extraction_status": empty_to_none(
            viz_row.get("viz_extraction_status")
        ),
        "viz_extracted_field_count": parse_optional_int(
            viz_row.get("extracted_field_count")
        ),
        "viz_selected_variant": empty_to_none(viz_row.get("viz_selected_variant")),
        "doi_status": empty_to_none(doi_row.get("status")),
        "doi_method": empty_to_none(doi_row.get("method")),
        "doi_score": parse_optional_float(doi_row.get("score")),
        "doi_score_margin": parse_optional_float(doi_row.get("score_margin")),
        "doi_label_text": empty_to_none(doi_row.get("label_text")),
        "doi_candidate_text": empty_to_none(doi_row.get("candidate_text")),
    }


def flatten_record_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    scalar_keys = (
        "sample_id",
        "filename",
        "relative_path",
        "generated_filename",
        "passport_stage_status",
        "passport_gate_status",
        "final_status",
        "coverage_status",
        "quality_status",
        "quality_score",
        "extraction_mode",
        "extracted_field_count",
        "review_required",
        "consistency_issue_count",
        "consistency_high_count",
        "consistency_medium_count",
        "consistency_low_count",
        "passport_number",
        "surname",
        "given_names",
        "nationality",
        "date_of_birth",
        "sex",
        "date_of_expiry",
        "date_of_issue",
        "document_type",
        "issuing_country",
        "personal_number",
        "mrz_available",
        "mrz_parse_status",
        "mrz_parse_mode",
        "mrz_ocr_variant",
        "mrz_ocr_confidence",
        "mrz_validation_status",
        "passport_number_check_valid",
        "birth_date_check_valid",
        "expiry_date_check_valid",
        "personal_number_check_valid",
        "final_check_valid",
        "all_main_checks_valid",
        "mrz_line_1",
        "mrz_line_2",
        "viz_available",
        "viz_extraction_status",
        "viz_extracted_field_count",
        "viz_selected_variant",
        "doi_status",
        "doi_method",
        "doi_score",
        "doi_score_margin",
        "doi_label_text",
        "doi_candidate_text",
    )

    row = {key: record.get(key) for key in scalar_keys}
    row["missing_fields"] = " | ".join(record["missing_fields"])
    row["review_reasons"] = " | ".join(record["review_reasons"])
    row["source_conflict_fields"] = " | ".join(record["source_conflict_fields"])
    row["consistency_issue_codes"] = " | ".join(
        issue["code"] for issue in record["consistency_issues"]
    )

    for field_name in CORE_FIELDS:
        row[f"{field_name}_source"] = record["field_sources"].get(field_name)
        row[f"{field_name}_quality"] = record["field_quality"].get(field_name)

    return row


def write_json(records: list[dict[str, Any]]) -> None:
    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def write_csv(records: list[dict[str, Any]]) -> None:
    rows = [flatten_record_for_csv(record) for record in records]
    if not rows:
        return

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(MANIFEST_CSV)
    processing_by_id = index_by_sample_id(load_csv(PROCESSING_CSV, required=False))
    mrz_by_id = index_by_sample_id(load_csv(MRZ_VALIDATED_CSV, required=False))
    viz_by_id = index_by_sample_id(load_csv(VIZ_FIELDS_CSV, required=False))
    doi_by_id = index_by_sample_id(load_csv(DOI_CSV, required=False))

    records = [
        build_final_record(
            manifest_row=item,
            processing_row=processing_by_id.get(item["sample_id"]),
            mrz_row=mrz_by_id.get(item["sample_id"]),
            viz_row=viz_by_id.get(item["sample_id"]),
            doi_row=doi_by_id.get(item["sample_id"]),
        )
        for item in manifest
    ]

    write_json(records)
    write_csv(records)

    coverage_counts = {
        status: sum(record["coverage_status"] == status for record in records)
        for status in ("complete", "partial", "failed")
    }
    quality_counts = {
        status: sum(record["quality_status"] == status for record in records)
        for status in ("high_confidence", "medium_confidence", "review", "low_confidence")
    }
    review = sum(bool(record["review_required"]) for record in records)
    conflicts = sum(bool(record["source_conflict_fields"]) for record in records)

    print("=" * 76)
    print("FINAL PASSPORT EXTRACTION - COVERAGE + QUALITY + CONSISTENCY")
    print("=" * 76)
    print(f"Input manifest records : {len(records)}")
    print("\nCoverage:")
    for status, count in coverage_counts.items():
        print(f"  {status:<18}: {count}")
    print("\nQuality:")
    for status, count in quality_counts.items():
        print(f"  {status:<18}: {count}")
    print(f"\nSource conflicts       : {conflicts}")
    print(f"Review required        : {review}")
    print(f"\nCSV : {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
