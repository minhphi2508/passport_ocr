from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_CSV = PROJECT_ROOT / "outputs" / "final_results" / "passport_extraction_results.csv"
DEFAULT_GT_CSV = PROJECT_ROOT / "ground_truth" / "passport_ground_truth.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "failure_audit"
DETAIL_CSV = OUTPUT_DIR / "failure_audit_details.csv"
SUMMARY_CSV = OUTPUT_DIR / "failure_audit_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "failure_audit_summary.json"

FIELDS = [
    "passport_number",
    "surname",
    "given_names",
    "nationality",
    "date_of_birth",
    "sex",
    "date_of_expiry",
    "date_of_issue",
]

MRZ_NATIVE_FIELD = {
    "passport_number": "passport_number",
    "surname": "surname",
    "given_names": "given_names",
    "nationality": "nationality",
    "date_of_birth": "birth_date",
    "sex": "sex",
    "date_of_expiry": "expiry_date",
}

CHECK_FIELD = {
    "passport_number": "passport_number_check_valid",
    "date_of_birth": "birth_date_check_valid",
    "date_of_expiry": "expiry_date_check_valid",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file:\n{path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def available(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"none", "null", "nan"}


def norm(field: str, value: Any) -> str | None:
    if not available(value):
        return None
    text = " ".join(str(value).strip().split())
    if field in {"passport_number", "surname", "given_names", "nationality", "sex"}:
        return text.upper()
    return text


def parse_bool(value: Any) -> bool | None:
    if not available(value):
        return None
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def load_optional_index(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = load_csv(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            filename = (row.get("filename") or "").strip()
            sample_id = Path(filename).stem if filename else ""
        if sample_id:
            result[sample_id] = row
    return result


def source_value(
    field: str,
    source_prefix: str,
    mrz_row: dict[str, str],
    viz_row: dict[str, str],
    doi_row: dict[str, str],
) -> str | None:
    if source_prefix.startswith("mrz") and field in MRZ_NATIVE_FIELD:
        return norm(field, mrz_row.get(MRZ_NATIVE_FIELD[field]))
    if source_prefix.startswith("viz"):
        if field == "date_of_issue":
            return norm(field, doi_row.get("date_of_issue"))
        return norm(field, viz_row.get(field))
    return None


def classify_error(
    field: str,
    expected: str,
    predicted: str | None,
    final_row: dict[str, str],
    processing_row: dict[str, str],
    mrz_row: dict[str, str],
    viz_row: dict[str, str],
    doi_row: dict[str, str],
) -> tuple[str, str]:
    page_status = (final_row.get("passport_stage_status") or processing_row.get("status") or "").strip()
    if page_status in {"no_passport_evidence", "image_read_failed", "unexpected_error", "empty_crop"}:
        return "passport_stage", f"passport_stage_status={page_status}"

    if field == "date_of_issue":
        doi_value = norm(field, doi_row.get("date_of_issue"))
        if doi_value == expected and predicted != expected:
            return "fusion_or_finalization", "DOI stage had correct value but final value differs"
        if not available(doi_row.get("date_of_issue")):
            return "doi_extraction", "date_of_issue missing from DOI stage"
        return "doi_extraction", f"DOI predicted {doi_value!r}, expected {expected!r}"

    source = (final_row.get(f"{field}_source") or "").strip()
    mrz_value = norm(field, mrz_row.get(MRZ_NATIVE_FIELD.get(field, "")))
    viz_value = norm(field, viz_row.get(field))

    # If an alternative source was already correct, the problem is source selection/fusion.
    if source.startswith("mrz") and viz_value == expected and mrz_value != expected:
        return "fusion", "VIZ had correct value but MRZ was selected"
    if source.startswith("viz") and mrz_value == expected and viz_value != expected:
        return "fusion", "MRZ had correct value but VIZ was selected"

    if source.startswith("mrz") or (not source and mrz_value is not None):
        check_name = CHECK_FIELD.get(field)
        check_value = parse_bool(mrz_row.get(check_name)) if check_name else None
        parse_status = (mrz_row.get("parse_status") or final_row.get("mrz_parse_status") or "").strip()
        parse_mode = (mrz_row.get("parse_mode") or final_row.get("mrz_parse_mode") or "").strip()
        if check_value is False:
            return "mrz_ocr", f"MRZ checksum failed for {field}"
        if parse_status != "success" or parse_mode == "padded_or_truncated":
            return "mrz_parse", f"parse_status={parse_status}, parse_mode={parse_mode}"
        return "mrz_ocr_or_parse", f"MRZ value={mrz_value!r}, expected={expected!r}"

    if source.startswith("viz") or (not source and viz_value is not None):
        extraction_status = (viz_row.get("viz_extraction_status") or final_row.get("viz_extraction_status") or "").strip()
        return "viz_field_extraction", f"viz_status={extraction_status}, value={viz_value!r}, expected={expected!r}"

    if predicted is None:
        mrz_available = str(final_row.get("mrz_available", "")).lower() == "true"
        viz_available = str(final_row.get("viz_available", "")).lower() == "true"
        if not mrz_available and not viz_available:
            return "upstream_missing", "Neither MRZ nor VIZ result available"
        return "field_missing", "Field unavailable despite upstream result(s)"

    return "unclassified", f"predicted={predicted!r}, expected={expected!r}, source={source!r}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify Passport OCR ground-truth failures by probable pipeline stage.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT_CSV)
    args = parser.parse_args()

    final_rows = load_csv(FINAL_CSV)
    gt_rows = load_csv(args.ground_truth)

    final_by_id = {row.get("sample_id", ""): row for row in final_rows if row.get("sample_id")}
    gt_by_id = {row.get("sample_id", ""): row for row in gt_rows if row.get("sample_id")}

    processing = load_optional_index(PROJECT_ROOT / "outputs" / "passport_pages_safe" / "processing_results.csv")
    mrz = load_optional_index(PROJECT_ROOT / "outputs" / "mrz_validated" / "mrz_validated_results.csv")
    viz = load_optional_index(PROJECT_ROOT / "outputs" / "viz_fields" / "viz_fields_results.csv")
    doi = load_optional_index(PROJECT_ROOT / "outputs" / "date_of_issue_hybrid_v3" / "date_of_issue_hybrid_v3_results.csv")

    common = sorted(set(final_by_id) & set(gt_by_id))
    if not common:
        raise RuntimeError("Không có sample_id chung giữa ground truth và final results.")

    detail_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    identity_category: dict[str, Counter[str]] = defaultdict(Counter)

    for sample_id in common:
        final_row = final_by_id[sample_id]
        gt_row = gt_by_id[sample_id]
        identity_id = (gt_row.get("identity_id") or "").strip() or sample_id

        for field in FIELDS:
            expected = norm(field, gt_row.get(field))
            if expected is None:
                continue
            predicted = norm(field, final_row.get(field))
            if predicted == expected:
                continue

            category, reason = classify_error(
                field=field,
                expected=expected,
                predicted=predicted,
                final_row=final_row,
                processing_row=processing.get(sample_id, {}),
                mrz_row=mrz.get(sample_id, {}),
                viz_row=viz.get(sample_id, {}),
                doi_row=doi.get(sample_id, {}),
            )
            category_counts[category] += 1
            field_counts[field] += 1
            identity_category[identity_id][category] += 1

            detail_rows.append({
                "sample_id": sample_id,
                "identity_id": identity_id,
                "split": (gt_row.get("split") or "unspecified").strip() or "unspecified",
                "filename": final_row.get("filename"),
                "field": field,
                "expected": expected,
                "predicted": predicted,
                "selected_source": final_row.get(f"{field}_source"),
                "selected_quality": final_row.get(f"{field}_quality"),
                "failure_category": category,
                "failure_reason": reason,
                "coverage_status": final_row.get("coverage_status"),
                "quality_status": final_row.get("quality_status"),
                "review_required": final_row.get("review_required"),
                "mrz_parse_status": final_row.get("mrz_parse_status"),
                "mrz_parse_mode": final_row.get("mrz_parse_mode"),
                "all_main_checks_valid": final_row.get("all_main_checks_valid"),
                "viz_extraction_status": final_row.get("viz_extraction_status"),
                "doi_status": final_row.get("doi_status"),
                "source_conflict_fields": final_row.get("source_conflict_fields"),
                "consistency_issue_codes": final_row.get("consistency_issue_codes"),
            })

    summary_rows: list[dict[str, Any]] = []
    total_errors = len(detail_rows)
    for category, count in category_counts.most_common():
        summary_rows.append({
            "dimension": "failure_category",
            "name": category,
            "count": count,
            "rate_of_all_field_errors": count / total_errors if total_errors else 0.0,
        })
    split_category_counts: Counter[tuple[str, str]] = Counter(
        (row["split"], row["failure_category"]) for row in detail_rows
    )
    for (split, category), count in sorted(split_category_counts.items()):
        summary_rows.append({
            "dimension": f"split:{split}:failure_category",
            "name": category,
            "count": count,
            "rate_of_all_field_errors": count / total_errors if total_errors else 0.0,
        })
    for field, count in field_counts.most_common():
        summary_rows.append({
            "dimension": "field",
            "name": field,
            "count": count,
            "rate_of_all_field_errors": count / total_errors if total_errors else 0.0,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(DETAIL_CSV, detail_rows)
    write_csv(SUMMARY_CSV, summary_rows)

    summary_json = {
        "matched_samples": len(common),
        "field_error_count": total_errors,
        "failure_categories": dict(category_counts.most_common()),
        "field_errors": dict(field_counts.most_common()),
        "identities_with_errors": len(identity_category),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary_json, file, ensure_ascii=False, indent=2)

    print("=" * 76)
    print("FAILURE AUDIT")
    print("=" * 76)
    print(f"Matched samples    : {len(common)}")
    print(f"Wrong field values : {total_errors}")
    print(f"Affected identities: {len(identity_category)}")
    print("\nProbable failure categories:")
    for category, count in category_counts.most_common():
        rate = count / total_errors if total_errors else 0.0
        print(f"  {category:<24}: {count:>5} ({rate:.1%})")
    print(f"\nDetails: {DETAIL_CSV}")
    print(f"Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
