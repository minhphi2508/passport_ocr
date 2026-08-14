from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_CSV = PROJECT_ROOT / "outputs" / "final_results" / "passport_extraction_results.csv"
GT_DIR = PROJECT_ROOT / "ground_truth"
DEFAULT_GT = GT_DIR / "passport_ground_truth.csv"

FIELDS = [
    "passport_number", "surname", "given_names", "nationality",
    "date_of_birth", "sex", "date_of_expiry", "date_of_issue",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file:\n{path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Không có row để ghi.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_template(output: Path) -> None:
    rows = load_csv(FINAL_CSV)
    template = []
    for row in rows:
        item: dict[str, Any] = {
            "sample_id": row.get("sample_id", ""),
            "identity_id": "",
            "split": "",
            "filename": row.get("filename", ""),
            "relative_path": row.get("relative_path", ""),
        }
        for field in FIELDS:
            item[field] = ""
        item["notes"] = ""
        template.append(item)
    write_csv(output, template)
    print(f"Ground-truth template: {output}")
    print("Điền identity_id + field chuẩn. split có thể để trống và dùng --assign-splits sau.")


def deterministic_identity_order(identity_ids: list[str], seed: int) -> list[str]:
    # Independent from Python hash randomization.
    return sorted(
        identity_ids,
        key=lambda identity: hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest(),
    )


def assign_splits(path: Path, output: Path, train: float, val: float, test: float, seed: int) -> None:
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError("train + val + test phải bằng 1.0")

    rows = load_csv(path)
    by_identity: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        identity = (row.get("identity_id") or "").strip()
        if not identity:
            raise ValueError(
                f"sample_id={row.get('sample_id')} chưa có identity_id. "
                "Phải gán identity_id trước để tránh leakage."
            )
        by_identity[identity].append(row)

    identities = deterministic_identity_order(list(by_identity), seed)
    n = len(identities)
    n_train = round(n * train)
    n_val = round(n * val)
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    split_by_identity: dict[str, str] = {}
    for i, identity in enumerate(identities):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        split_by_identity[identity] = split

    for row in rows:
        row["split"] = split_by_identity[row["identity_id"].strip()]

    write_csv(output, rows)
    counts = defaultdict(int)
    for identity, split in split_by_identity.items():
        counts[split] += 1
    print(f"Saved: {output}")
    print(f"Identity splits: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print("Không có identity nào xuất hiện ở nhiều split.")


def validate_ground_truth(path: Path) -> None:
    rows = load_csv(path)
    seen_samples: set[str] = set()
    identity_splits: dict[str, set[str]] = defaultdict(set)
    problems: list[str] = []

    for index, row in enumerate(rows, start=2):
        sample_id = (row.get("sample_id") or "").strip()
        identity = (row.get("identity_id") or "").strip()
        split = (row.get("split") or "").strip()
        if not sample_id:
            problems.append(f"line {index}: missing sample_id")
        elif sample_id in seen_samples:
            problems.append(f"line {index}: duplicate sample_id={sample_id}")
        else:
            seen_samples.add(sample_id)
        if not identity:
            problems.append(f"line {index}: missing identity_id")
        if identity and split:
            identity_splits[identity].add(split)

    for identity, splits in identity_splits.items():
        if len(splits) > 1:
            problems.append(f"identity leakage: {identity} appears in {sorted(splits)}")

    annotated = 0
    field_cells = 0
    for row in rows:
        values = [str(row.get(field, "")).strip() for field in FIELDS]
        annotated += int(any(values))
        field_cells += sum(bool(value) for value in values)

    print("=" * 76)
    print("GROUND TRUTH VALIDATION")
    print("=" * 76)
    print(f"Rows             : {len(rows)}")
    print(f"Annotated samples: {annotated}")
    print(f"Annotated fields : {field_cells}/{len(rows) * len(FIELDS)}")
    print(f"Identities       : {len({(r.get('identity_id') or '').strip() for r in rows if (r.get('identity_id') or '').strip()})}")
    if problems:
        print("\nProblems:")
        for problem in problems[:50]:
            print(f"  - {problem}")
        raise SystemExit(2)
    print("\n✓ Ground truth structure is valid.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ground-truth utilities with identity-safe splitting.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create-template")
    p_create.add_argument("--output", type=Path, default=DEFAULT_GT)

    p_split = sub.add_parser("assign-splits")
    p_split.add_argument("--input", type=Path, default=DEFAULT_GT)
    p_split.add_argument("--output", type=Path, default=DEFAULT_GT)
    p_split.add_argument("--train", type=float, default=0.70)
    p_split.add_argument("--val", type=float, default=0.15)
    p_split.add_argument("--test", type=float, default=0.15)
    p_split.add_argument("--seed", type=int, default=42)

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("--input", type=Path, default=DEFAULT_GT)

    args = parser.parse_args()
    if args.command == "create-template":
        create_template(args.output)
    elif args.command == "assign-splits":
        assign_splits(args.input, args.output, args.train, args.val, args.test, args.seed)
    elif args.command == "validate":
        validate_ground_truth(args.input)


if __name__ == "__main__":
    main()
