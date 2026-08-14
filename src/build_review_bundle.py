from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_CSV = PROJECT_ROOT / "outputs" / "evaluation" / "failure_audit" / "failure_audit_details.csv"
FINAL_CSV = PROJECT_ROOT / "outputs" / "final_results" / "passport_extraction_results.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "evaluation" / "review_bundle"


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file:\n{path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def find_source_image(final_row: dict[str, str]) -> Path | None:
    rel = (final_row.get("relative_path") or "").strip()
    if rel:
        path = PROJECT_ROOT / "input_images" / Path(rel)
        if path.exists():
            return path
    filename = (final_row.get("filename") or "").strip()
    if filename:
        candidates = list((PROJECT_ROOT / "input_images").rglob(filename))
        if candidates:
            return candidates[0]
    return None


def candidate_artifacts(sample_id: str) -> list[tuple[str, Path]]:
    return [
        ("passport_page", PROJECT_ROOT / "outputs" / "passport_pages_safe" / "transformed" / f"{sample_id}.jpg"),
        ("passport_page_alt", PROJECT_ROOT / "outputs" / "passport_pages_safe" / "passport_pages" / f"{sample_id}.jpg"),
        ("mrz_original", PROJECT_ROOT / "outputs" / "mrz_stage" / "original" / f"{sample_id}.jpg"),
        ("mrz_grayscale", PROJECT_ROOT / "outputs" / "mrz_stage" / "grayscale" / f"{sample_id}.jpg"),
        ("mrz_threshold", PROJECT_ROOT / "outputs" / "mrz_stage" / "threshold" / f"{sample_id}.jpg"),
        ("viz_color", PROJECT_ROOT / "outputs" / "viz_stage" / "color" / f"{sample_id}.jpg"),
        ("viz_enhanced", PROJECT_ROOT / "outputs" / "viz_stage" / "enhanced" / f"{sample_id}.jpg"),
        ("viz_grayscale", PROJECT_ROOT / "outputs" / "viz_stage" / "grayscale" / f"{sample_id}.jpg"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact review folder for GT failures.")
    parser.add_argument("--audit", type=Path, default=AUDIT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--category", action="append", default=[], help="Only include selected failure category; repeatable.")
    args = parser.parse_args()

    audit_rows = load_csv(args.audit)
    final_rows = load_csv(FINAL_CSV)
    final_by_id = {row.get("sample_id", ""): row for row in final_rows if row.get("sample_id")}

    if args.category:
        wanted = set(args.category)
        audit_rows = [row for row in audit_rows if row.get("failure_category") in wanted]

    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit_rows:
        sample_id = (row.get("sample_id") or "").strip()
        if sample_id:
            by_sample[sample_id].append(row)

    selected_ids = sorted(
        by_sample,
        key=lambda sample_id: (
            -len(by_sample[sample_id]),
            sample_id,
        ),
    )[: max(0, args.max_samples)]

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []

    for rank, sample_id in enumerate(selected_ids, start=1):
        errors = by_sample[sample_id]
        final_row = final_by_id.get(sample_id, {})
        categories = sorted({row.get("failure_category", "") for row in errors})
        sample_dir = args.output / f"{rank:04d}__{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        copied: list[str] = []
        src = find_source_image(final_row)
        if src is not None:
            ext = src.suffix.lower() or ".jpg"
            if safe_copy(src, sample_dir / f"00_input{ext}"):
                copied.append("input")

        for order, (label, path) in enumerate(candidate_artifacts(sample_id), start=1):
            if safe_copy(path, sample_dir / f"{order:02d}_{label}{path.suffix}"):
                copied.append(label)

        metadata = {
            "sample_id": sample_id,
            "filename": final_row.get("filename"),
            "relative_path": final_row.get("relative_path"),
            "coverage_status": final_row.get("coverage_status"),
            "quality_status": final_row.get("quality_status"),
            "review_reasons": final_row.get("review_reasons"),
            "failure_categories": categories,
            "errors": errors,
            "copied_artifacts": copied,
        }
        with (sample_dir / "review.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        lines = [
            f"sample_id: {sample_id}",
            f"filename: {final_row.get('filename', '')}",
            f"quality_status: {final_row.get('quality_status', '')}",
            f"failure_categories: {' | '.join(categories)}",
            "",
            "FIELD ERRORS",
        ]
        for row in errors:
            lines.extend([
                f"- {row.get('field')}: expected={row.get('expected')!r}, predicted={row.get('predicted')!r}",
                f"  category={row.get('failure_category')} source={row.get('selected_source')} reason={row.get('failure_reason')}",
            ])
        (sample_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")

        index_rows.append({
            "rank": rank,
            "sample_id": sample_id,
            "filename": final_row.get("filename"),
            "error_field_count": len(errors),
            "failure_categories": " | ".join(categories),
            "quality_status": final_row.get("quality_status"),
            "copied_artifacts": " | ".join(copied),
        })

    if index_rows:
        with (args.output / "review_index.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)

    print("=" * 76)
    print("FAILURE REVIEW BUNDLE")
    print("=" * 76)
    print(f"Failure samples available: {len(by_sample)}")
    print(f"Samples bundled          : {len(selected_ids)}")
    print(f"Output                   : {args.output}")
    print("Mỗi sample có input/crop khả dụng + README.txt + review.json.")


if __name__ == "__main__":
    main()
