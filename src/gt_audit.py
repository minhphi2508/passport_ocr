from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GT = PROJECT_ROOT / "ground_truth" / "passport_ground_truth.csv"
FINAL_CSV = PROJECT_ROOT / "outputs" / "final_results" / "passport_extraction_results.csv"
DETAILS_CSV = PROJECT_ROOT / "outputs" / "evaluation" / "ground_truth_sample_details.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "gt_audit"
AUDIT_CSV = OUTPUT_DIR / "gt_audit_details.csv"
SUMMARY_CSV = OUTPUT_DIR / "gt_audit_summary.csv"
REVIEW_CSV = OUTPUT_DIR / "gt_review_queue.csv"

FIELDS = (
    "passport_number",
    "surname",
    "given_names",
    "nationality",
    "date_of_birth",
    "sex",
    "date_of_expiry",
    "date_of_issue",
)


def clean(value: Any) -> str:
    return str(value or "").strip()


def norm_text(value: Any) -> str:
    return " ".join(clean(value).upper().split())


def parse_iso(value: Any) -> date | None:
    text = clean(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def row_key(row: dict[str, str]) -> str:
    return clean(row.get("sample_id") or row.get("filename"))


def swap_day_month_candidate(value: str) -> str | None:
    d = parse_iso(value)
    if d is None:
        return None
    # YYYY-MM-DD -> YYYY-DD-MM, only if both are valid and differ.
    if d.day > 12 or d.month > 12 or d.day == d.month:
        return None
    try:
        swapped = date(d.year, d.day, d.month)
    except ValueError:
        return None
    if swapped == d:
        return None
    return swapped.isoformat()


def temporal_flags(row: dict[str, str]) -> list[dict[str, str]]:
    out = []
    dob = parse_iso(row.get("date_of_birth"))
    doi = parse_iso(row.get("date_of_issue"))
    exp = parse_iso(row.get("date_of_expiry"))
    today = date.today()

    if dob and doi and doi <= dob:
        out.append({
            "flag": "doi_not_after_dob",
            "field": "date_of_issue",
            "severity": "high",
            "message": f"DOI {doi} <= DOB {dob}",
        })

    if doi and exp and doi >= exp:
        out.append({
            "flag": "doi_not_before_expiry",
            "field": "date_of_issue",
            "severity": "high",
            "message": f"DOI {doi} >= expiry {exp}",
        })

    if dob and exp and exp <= dob:
        out.append({
            "flag": "expiry_not_after_dob",
            "field": "date_of_expiry",
            "severity": "high",
            "message": f"expiry {exp} <= DOB {dob}",
        })

    if dob and dob > today:
        out.append({
            "flag": "dob_in_future",
            "field": "date_of_birth",
            "severity": "high",
            "message": f"DOB {dob} is in future",
        })

    if doi and doi > today:
        out.append({
            "flag": "doi_in_future",
            "field": "date_of_issue",
            "severity": "medium",
            "message": f"DOI {doi} is in future",
        })

    if doi and exp:
        years = (exp - doi).days / 365.2425
        if years <= 0:
            pass
        elif years < 0.25 or years > 15.5:
            out.append({
                "flag": "unusual_validity_period",
                "field": "date_of_issue",
                "severity": "medium",
                "message": f"passport validity period ~{years:.2f} years",
            })

    return out


def identity_consistency_flags(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_identity: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        identity = clean(row.get("identity_id"))
        if identity:
            by_identity[identity].append(row)

    results = []
    for identity, group in by_identity.items():
        for field in FIELDS:
            values = [
                norm_text(r.get(field))
                for r in group
                if clean(r.get(field))
            ]
            unique = sorted(set(values))
            if len(unique) <= 1:
                continue

            for r in group:
                results.append({
                    "sample_id": row_key(r),
                    "filename": clean(r.get("filename")),
                    "identity_id": identity,
                    "flag": "identity_field_inconsistent",
                    "field": field,
                    "severity": "high",
                    "message": f"{field} has multiple GT values in identity: {unique}",
                    "gt_value": clean(r.get(field)),
                    "prediction_value": "",
                    "suggested_review_value": "",
                })

    return results


def prediction_disagreement_flags(
    gt_rows: list[dict[str, str]],
    prediction_map: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    results = []

    for row in gt_rows:
        key = row_key(row)
        pred = prediction_map.get(key) or {}
        for field in FIELDS:
            gt = norm_text(row.get(field))
            pv = norm_text(pred.get(field))
            if not gt or not pv or gt == pv:
                continue

            severity = "low"
            flag = "gt_differs_from_prediction"
            suggestion = ""

            if field in {"date_of_birth", "date_of_expiry", "date_of_issue"}:
                swapped = swap_day_month_candidate(clean(row.get(field)))
                if swapped and swapped == clean(pred.get(field)):
                    severity = "high"
                    flag = "possible_day_month_swap"
                    suggestion = swapped

            results.append({
                "sample_id": key,
                "filename": clean(row.get("filename")),
                "identity_id": clean(row.get("identity_id")),
                "flag": flag,
                "field": field,
                "severity": severity,
                "message": f"GT={clean(row.get(field))} vs prediction={clean(pred.get(field))}",
                "gt_value": clean(row.get(field)),
                "prediction_value": clean(pred.get(field)),
                "suggested_review_value": suggestion,
            })

    return results


def duplicate_sample_flags(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    counter = Counter(row_key(r) for r in rows if row_key(r))
    results = []
    for row in rows:
        key = row_key(row)
        if key and counter[key] > 1:
            results.append({
                "sample_id": key,
                "filename": clean(row.get("filename")),
                "identity_id": clean(row.get("identity_id")),
                "flag": "duplicate_sample",
                "field": "",
                "severity": "high",
                "message": f"sample appears {counter[key]} times in GT",
                "gt_value": "",
                "prediction_value": "",
                "suggested_review_value": "",
            })
    return results


def blank_field_flags(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    results = []
    for row in rows:
        for field in FIELDS:
            if clean(row.get(field)):
                continue
            results.append({
                "sample_id": row_key(row),
                "filename": clean(row.get("filename")),
                "identity_id": clean(row.get("identity_id")),
                "flag": "blank_gt_field",
                "field": field,
                "severity": "medium",
                "message": f"GT field {field} is blank",
                "gt_value": "",
                "prediction_value": "",
                "suggested_review_value": "",
            })
    return results


def temporal_row_flags(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    results = []
    for row in rows:
        for item in temporal_flags(row):
            results.append({
                "sample_id": row_key(row),
                "filename": clean(row.get("filename")),
                "identity_id": clean(row.get("identity_id")),
                **item,
                "gt_value": clean(row.get(item["field"])),
                "prediction_value": "",
                "suggested_review_value": "",
            })
    return results


def build_prediction_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = load_csv(path)
    result = {}
    for row in rows:
        key = row_key(row)
        if key:
            result[key] = row
    return result


def audit(gt_path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    gt_rows = load_csv(gt_path)
    prediction_map = build_prediction_map(FINAL_CSV)

    flags: list[dict[str, str]] = []
    flags.extend(duplicate_sample_flags(gt_rows))
    flags.extend(blank_field_flags(gt_rows))
    flags.extend(temporal_row_flags(gt_rows))
    flags.extend(identity_consistency_flags(gt_rows))
    flags.extend(prediction_disagreement_flags(gt_rows, prediction_map))

    # Deduplicate exact audit rows.
    unique = []
    seen = set()
    for row in flags:
        key = (
            row["sample_id"],
            row["flag"],
            row["field"],
            row["message"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(row)

    priority = {"high": 0, "medium": 1, "low": 2}
    unique.sort(
        key=lambda r: (
            priority.get(r["severity"], 9),
            r["identity_id"],
            r["sample_id"],
            r["field"],
        )
    )

    summary = Counter(r["flag"] for r in unique)
    return unique, dict(summary)


def write_reports(
    rows: list[dict[str, str]],
    summary: dict[str, int],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fields = [
        "sample_id",
        "filename",
        "identity_id",
        "severity",
        "flag",
        "field",
        "gt_value",
        "prediction_value",
        "suggested_review_value",
        "message",
    ]

    with AUDIT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary_rows = [
        {"flag": flag, "count": count}
        for flag, count in sorted(
            summary.items(),
            key=lambda x: (-x[1], x[0]),
        )
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["flag", "count"])
        w.writeheader()
        w.writerows(summary_rows)

    # Human review queue: only high/medium flags, grouped by sample.
    review_rows = [
        r for r in rows
        if r["severity"] in {"high", "medium"}
    ]
    with REVIEW_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(review_rows)


def print_summary(
    gt_path: Path,
    rows: list[dict[str, str]],
    summary: dict[str, int],
) -> None:
    severity = Counter(r["severity"] for r in rows)
    samples = len(set(r["sample_id"] for r in rows if r["sample_id"]))

    print("=" * 76)
    print("GROUND TRUTH AUDIT")
    print("=" * 76)
    print(f"GT file             : {gt_path}")
    print(f"Flagged audit rows  : {len(rows)}")
    print(f"Flagged samples     : {samples}")
    print(f"High severity       : {severity.get('high', 0)}")
    print(f"Medium severity     : {severity.get('medium', 0)}")
    print(f"Low severity        : {severity.get('low', 0)}")
    print()
    print("Flag counts:")
    for flag, count in sorted(summary.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {flag:<30}: {count}")
    print()
    print(f"Details : {AUDIT_CSV}")
    print(f"Summary : {SUMMARY_CSV}")
    print(f"Review  : {REVIEW_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GT,
    )
    args = parser.parse_args()

    if not args.ground_truth.exists():
        raise FileNotFoundError(args.ground_truth)

    rows, summary = audit(args.ground_truth)
    write_reports(rows, summary)
    print_summary(args.ground_truth, rows, summary)


if __name__ == "__main__":
    main()
