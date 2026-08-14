from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_CSV = PROJECT_ROOT / "outputs" / "final_results" / "passport_extraction_results.csv"
DEFAULT_GT = PROJECT_ROOT / "ground_truth" / "passport_ground_truth.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_CSV = OUTPUT_DIR / "split_metrics.csv"

FIELDS = [
    "passport_number", "surname", "given_names", "nationality",
    "date_of_birth", "sex", "date_of_expiry", "date_of_issue",
]


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


def majority(values: list[str]) -> str | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report image-level and identity-level GT accuracy separately for each split.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    args = parser.parse_args()

    final = {r.get("sample_id", ""): r for r in load_csv(FINAL_CSV) if r.get("sample_id")}
    gt_rows = load_csv(args.ground_truth)
    gt_rows = [r for r in gt_rows if r.get("sample_id") in final]
    if not gt_rows:
        raise RuntimeError("Không có ground-truth sample khớp final results.")

    splits = sorted({(r.get("split") or "unspecified").strip() or "unspecified" for r in gt_rows})
    output: list[dict[str, Any]] = []

    for split in splits:
        rows = [r for r in gt_rows if ((r.get("split") or "unspecified").strip() or "unspecified") == split]
        image_field: dict[str, list[bool]] = defaultdict(list)
        image_all: list[bool] = []
        identities: dict[str, list[dict[str, str]]] = defaultdict(list)

        for gt in rows:
            pred = final[gt["sample_id"]]
            identity = (gt.get("identity_id") or "").strip() or gt["sample_id"]
            identities[identity].append(gt)
            flags: list[bool] = []
            for field in FIELDS:
                expected = norm(field, gt.get(field))
                if expected is None:
                    continue
                correct = norm(field, pred.get(field)) == expected
                image_field[field].append(correct)
                flags.append(correct)
            if flags:
                image_all.append(all(flags))

        for field in FIELDS:
            values = image_field[field]
            if values:
                output.append({
                    "split": split, "level": "image", "metric": field,
                    "correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values),
                })
        if image_all:
            output.append({
                "split": split, "level": "image", "metric": "all_fields_correct",
                "correct": sum(image_all), "total": len(image_all), "accuracy": sum(image_all) / len(image_all),
            })

        identity_field: dict[str, list[bool]] = defaultdict(list)
        identity_all: list[bool] = []
        for identity, identity_gt_rows in identities.items():
            flags: list[bool] = []
            for field in FIELDS:
                expected_values = [norm(field, row.get(field)) for row in identity_gt_rows]
                expected_values = [v for v in expected_values if v is not None]
                if not expected_values:
                    continue
                expected = majority(expected_values)
                predicted_values = [
                    norm(field, final[row["sample_id"]].get(field)) for row in identity_gt_rows
                ]
                predicted_values = [v for v in predicted_values if v is not None]
                predicted = majority(predicted_values)
                correct = predicted == expected
                identity_field[field].append(correct)
                flags.append(correct)
            if flags:
                identity_all.append(all(flags))

        for field in FIELDS:
            values = identity_field[field]
            if values:
                output.append({
                    "split": split, "level": "identity", "metric": field,
                    "correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values),
                })
        if identity_all:
            output.append({
                "split": split, "level": "identity", "metric": "all_fields_correct",
                "correct": sum(identity_all), "total": len(identity_all), "accuracy": sum(identity_all) / len(identity_all),
            })

    write_csv(OUTPUT_CSV, output)
    print("=" * 76)
    print("SPLIT METRICS")
    print("=" * 76)
    for row in output:
        if row["metric"] == "all_fields_correct":
            print(
                f"{row['split']:<12} {row['level']:<9} all_fields_correct: "
                f"{row['correct']}/{row['total']} ({row['accuracy']:.1%})"
            )
    print(f"\nCSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
