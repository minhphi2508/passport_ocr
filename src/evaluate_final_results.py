from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_CSV = PROJECT_ROOT / "outputs" / "final_results" / "passport_extraction_results.csv"
DEFAULT_GT_CSV = PROJECT_ROOT / "ground_truth" / "passport_ground_truth.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"

SUMMARY_CSV = OUTPUT_DIR / "end_to_end_summary.csv"
DETAIL_CSV = OUTPUT_DIR / "end_to_end_details.csv"
GT_SAMPLE_DETAILS_CSV = OUTPUT_DIR / "ground_truth_sample_details.csv"
IDENTITY_DETAILS_CSV = OUTPUT_DIR / "identity_level_details.csv"
IDENTITY_SUMMARY_CSV = OUTPUT_DIR / "identity_level_summary.csv"
QUALITY_CALIBRATION_CSV = OUTPUT_DIR / "quality_calibration.csv"
GT_TEMPLATE_CSV = OUTPUT_DIR / "ground_truth_template.csv"

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

QUALITY_ORDER = (
    "high_confidence",
    "medium_confidence",
    "review",
    "low_confidence",
)


def load_rows(path: Path, required: bool = True) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Không thấy file:\n{path}")
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def is_available(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"none", "null", "nan"}


def normalize_value(field: str, value: Any) -> str | None:
    if not is_available(value):
        return None

    text = " ".join(str(value).strip().split())
    if field in {"surname", "given_names", "nationality", "sex", "passport_number"}:
        return text.upper()
    return text


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _int_value(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def availability_and_quality_evaluation(rows: list[dict[str, str]]) -> None:
    total = len(rows)
    if total == 0:
        raise RuntimeError("Final results đang rỗng.")

    coverage_counts = {
        status: sum(
            (row.get("coverage_status") or row.get("final_status")) == status
            for row in rows
        )
        for status in ("complete", "partial", "failed")
    }

    quality_counts = {
        status: sum(row.get("quality_status") == status for row in rows)
        for status in QUALITY_ORDER
    }

    review_count = sum(
        str(row.get("review_required", "")).strip().lower() == "true"
        for row in rows
    )
    source_conflict_count = sum(
        is_available(row.get("source_conflict_fields")) for row in rows
    )
    high_consistency_issue_records = sum(
        _int_value(row.get("consistency_high_count")) > 0 for row in rows
    )

    field_counts = {
        field: sum(is_available(row.get(field)) for row in rows)
        for field in FINAL_FIELDS
    }

    detail_rows: list[dict[str, Any]] = []
    for row in rows:
        missing_fields = [
            field for field in FINAL_FIELDS if not is_available(row.get(field))
        ]
        detail_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "filename": row.get("filename"),
                "coverage_status": row.get("coverage_status") or row.get("final_status"),
                "quality_status": row.get("quality_status"),
                "quality_score": row.get("quality_score"),
                "review_required": row.get("review_required"),
                "passport_stage_status": row.get("passport_stage_status"),
                "mrz_parse_status": row.get("mrz_parse_status"),
                "mrz_parse_mode": row.get("mrz_parse_mode"),
                "all_main_checks_valid": row.get("all_main_checks_valid"),
                "doi_status": row.get("doi_status"),
                "source_conflict_fields": row.get("source_conflict_fields"),
                "consistency_issue_codes": row.get("consistency_issue_codes"),
                "missing_field_count": len(missing_fields),
                "missing_fields": " | ".join(missing_fields),
            }
        )

    summary_rows: list[dict[str, Any]] = [
        {"metric": "total_images", "count": total, "rate": 1.0},
    ]

    for status, count in coverage_counts.items():
        summary_rows.append(
            {"metric": f"coverage__{status}", "count": count, "rate": count / total}
        )

    for status, count in quality_counts.items():
        summary_rows.append(
            {"metric": f"quality__{status}", "count": count, "rate": count / total}
        )

    summary_rows.extend(
        [
            {
                "metric": "review_required",
                "count": review_count,
                "rate": review_count / total,
            },
            {
                "metric": "source_conflict_records",
                "count": source_conflict_count,
                "rate": source_conflict_count / total,
            },
            {
                "metric": "high_consistency_issue_records",
                "count": high_consistency_issue_records,
                "rate": high_consistency_issue_records / total,
            },
        ]
    )

    for field in FINAL_FIELDS:
        summary_rows.append(
            {
                "metric": f"field_available__{field}",
                "count": field_counts[field],
                "rate": field_counts[field] / total,
            }
        )

    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(DETAIL_CSV, detail_rows)

    print("=" * 76)
    print("END-TO-END COVERAGE / QUALITY")
    print("=" * 76)
    print(f"Total images : {total}")
    print("\nCoverage:")
    for status, count in coverage_counts.items():
        print(f"  {status:<18}: {count}/{total} ({count / total:.1%})")
    print("\nQuality:")
    for status, count in quality_counts.items():
        print(f"  {status:<18}: {count}/{total} ({count / total:.1%})")
    print(f"\nReview required       : {review_count}/{total} ({review_count / total:.1%})")
    print(
        f"Source conflict records: {source_conflict_count}/{total} "
        f"({source_conflict_count / total:.1%})"
    )


def write_ground_truth_template(final_rows: list[dict[str, str]]) -> None:
    rows: list[dict[str, Any]] = []

    for row in final_rows:
        item: dict[str, Any] = {
            "sample_id": row.get("sample_id"),
            "identity_id": "",
            "filename": row.get("filename"),
        }
        for field in FINAL_FIELDS:
            item[field] = ""
        rows.append(item)

    write_csv(GT_TEMPLATE_CSV, rows)
    print(f"Ground-truth template: {GT_TEMPLATE_CSV}")


def majority_nonempty(values: list[str | None]) -> str | None:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def evaluate_with_ground_truth(
    final_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
) -> None:
    pred_by_id = {
        row.get("sample_id", ""): row
        for row in final_rows
        if row.get("sample_id")
    }
    gt_by_id = {
        row.get("sample_id", ""): row
        for row in gt_rows
        if row.get("sample_id")
    }

    common_ids = sorted(set(pred_by_id) & set(gt_by_id))
    if not common_ids:
        raise RuntimeError("Ground truth không có sample_id trùng với final results.")

    sample_details: list[dict[str, Any]] = []
    sample_field_correct: dict[str, list[bool]] = defaultdict(list)
    sample_all_correct_values: list[bool] = []

    for sample_id in common_ids:
        pred = pred_by_id[sample_id]
        gt = gt_by_id[sample_id]
        identity_id = (gt.get("identity_id") or "").strip() or sample_id

        detail: dict[str, Any] = {
            "sample_id": sample_id,
            "identity_id": identity_id,
            "filename": pred.get("filename"),
            "coverage_status": pred.get("coverage_status") or pred.get("final_status"),
            "quality_status": pred.get("quality_status"),
            "quality_score": pred.get("quality_score"),
        }

        comparable_count = 0
        correct_count = 0

        for field in FINAL_FIELDS:
            expected = normalize_value(field, gt.get(field))
            predicted = normalize_value(field, pred.get(field))

            if expected is None:
                detail[f"{field}_expected"] = None
                detail[f"{field}_predicted"] = predicted
                detail[f"{field}_correct"] = None
                continue

            comparable_count += 1
            correct = predicted == expected
            correct_count += int(correct)
            sample_field_correct[field].append(correct)

            detail[f"{field}_expected"] = expected
            detail[f"{field}_predicted"] = predicted
            detail[f"{field}_correct"] = correct

        all_correct = comparable_count > 0 and correct_count == comparable_count
        sample_all_correct_values.append(all_correct)
        detail["comparable_field_count"] = comparable_count
        detail["correct_field_count"] = correct_count
        detail["all_comparable_fields_correct"] = all_correct
        sample_details.append(detail)

    write_csv(GT_SAMPLE_DETAILS_CSV, sample_details)

    print("\n" + "=" * 76)
    print("GROUND-TRUTH SAMPLE-LEVEL ACCURACY")
    print("=" * 76)
    print(f"Matched samples : {len(common_ids)}")

    for field in FINAL_FIELDS:
        values = sample_field_correct[field]
        if values:
            accuracy = sum(values) / len(values)
            print(f"{field:<24}: {sum(values)}/{len(values)} ({accuracy:.1%})")

    if sample_all_correct_values:
        all_acc = sum(sample_all_correct_values) / len(sample_all_correct_values)
        print(
            f"all_fields_correct       : "
            f"{sum(sample_all_correct_values)}/{len(sample_all_correct_values)} "
            f"({all_acc:.1%})"
        )

    evaluate_identity_level(sample_details)
    evaluate_quality_calibration(sample_details)


def evaluate_quality_calibration(sample_details: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_details:
        grouped[str(row.get("quality_status") or "unknown")].append(row)

    rows: list[dict[str, Any]] = []
    order = list(QUALITY_ORDER) + ["unknown"]

    for quality_status in order:
        group = grouped.get(quality_status, [])
        if not group:
            continue

        all_correct_values = [
            bool(row["all_comparable_fields_correct"])
            for row in group
            if row.get("comparable_field_count", 0) > 0
        ]
        field_correct = sum(int(row.get("correct_field_count", 0)) for row in group)
        field_total = sum(int(row.get("comparable_field_count", 0)) for row in group)

        rows.append(
            {
                "quality_status": quality_status,
                "sample_count": len(group),
                "all_fields_correct_count": sum(all_correct_values),
                "all_fields_correct_rate": (
                    sum(all_correct_values) / len(all_correct_values)
                    if all_correct_values
                    else None
                ),
                "field_correct_count": field_correct,
                "field_total_count": field_total,
                "field_accuracy": field_correct / field_total if field_total else None,
            }
        )

    write_csv(QUALITY_CALIBRATION_CSV, rows)

    print("\nQuality calibration:")
    for row in rows:
        rate = row["all_fields_correct_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.1%}"
        print(
            f"  {row['quality_status']:<18}: "
            f"samples={row['sample_count']}, all-fields={rate_text}"
        )
    print(f"Quality calibration CSV: {QUALITY_CALIBRATION_CSV}")


def evaluate_identity_level(sample_details: list[dict[str, Any]]) -> None:
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_details:
        by_identity[str(row["identity_id"])].append(row)

    identity_rows: list[dict[str, Any]] = []
    identity_field_correct: dict[str, list[bool]] = defaultdict(list)
    identity_all_correct: list[bool] = []

    for identity_id, rows in sorted(by_identity.items()):
        output: dict[str, Any] = {
            "identity_id": identity_id,
            "sample_count": len(rows),
        }
        comparable_count = 0
        correct_count = 0

        for field in FINAL_FIELDS:
            expected_values = [
                row.get(f"{field}_expected")
                for row in rows
                if row.get(f"{field}_expected") is not None
            ]
            predicted_values = [
                row.get(f"{field}_predicted")
                for row in rows
                if row.get(f"{field}_predicted") is not None
            ]

            expected = majority_nonempty(expected_values)
            predicted = majority_nonempty(predicted_values)

            if expected is None:
                output[f"{field}_expected"] = None
                output[f"{field}_prediction_consensus"] = predicted
                output[f"{field}_correct"] = None
                continue

            comparable_count += 1
            correct = predicted == expected
            correct_count += int(correct)
            identity_field_correct[field].append(correct)

            output[f"{field}_expected"] = expected
            output[f"{field}_prediction_consensus"] = predicted
            output[f"{field}_correct"] = correct

        all_correct = comparable_count > 0 and correct_count == comparable_count
        identity_all_correct.append(all_correct)
        output["comparable_field_count"] = comparable_count
        output["correct_field_count"] = correct_count
        output["all_comparable_fields_correct"] = all_correct
        identity_rows.append(output)

    write_csv(IDENTITY_DETAILS_CSV, identity_rows)

    summary_rows: list[dict[str, Any]] = []
    for field in FINAL_FIELDS:
        values = identity_field_correct[field]
        if not values:
            continue
        summary_rows.append(
            {
                "metric": f"identity_accuracy__{field}",
                "count_correct": sum(values),
                "count_total": len(values),
                "accuracy": sum(values) / len(values),
            }
        )

    if identity_all_correct:
        summary_rows.append(
            {
                "metric": "identity_accuracy__all_fields_correct",
                "count_correct": sum(identity_all_correct),
                "count_total": len(identity_all_correct),
                "accuracy": sum(identity_all_correct) / len(identity_all_correct),
            }
        )

    write_csv(IDENTITY_SUMMARY_CSV, summary_rows)

    print("\n" + "=" * 76)
    print("IDENTITY-LEVEL ACCURACY (EQUAL WEIGHT PER PASSPORT IDENTITY)")
    print("=" * 76)
    print(f"Identities : {len(identity_rows)}")

    for row in summary_rows:
        print(
            f"{row['metric']:<42}: "
            f"{row['count_correct']}/{row['count_total']} "
            f"({row['accuracy']:.1%})"
        )

    print(f"\nSample GT details   : {GT_SAMPLE_DETAILS_CSV}")
    print(f"Identity details    : {IDENTITY_DETAILS_CSV}")
    print(f"Identity summary    : {IDENTITY_SUMMARY_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Passport OCR coverage, quality/consistency and optional "
            "ground-truth accuracy."
        )
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GT_CSV,
        help=(
            "Ground-truth CSV containing sample_id, identity_id and final fields. "
            "Default: ground_truth/passport_ground_truth.csv"
        ),
    )
    parser.add_argument(
        "--write-ground-truth-template",
        action="store_true",
        help="Create a GT template from current final results.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_rows = load_rows(FINAL_CSV)
    availability_and_quality_evaluation(final_rows)

    if args.write_ground_truth_template:
        write_ground_truth_template(final_rows)

    gt_rows = load_rows(args.ground_truth, required=False)

    if gt_rows:
        evaluate_with_ground_truth(final_rows, gt_rows)
    else:
        print("\nGround truth chưa có -> báo cáo coverage/quality, chưa có accuracy thật.")
        print(
            "Tạo template bằng: "
            "python src/evaluate_final_results.py --write-ground-truth-template"
        )
        print(
            "Sau đó điền identity_id + giá trị chuẩn và lưu thành: "
            f"{DEFAULT_GT_CSV}"
        )


if __name__ == "__main__":
    main()
