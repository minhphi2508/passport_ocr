from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from annotation_sampling import (
    FIELDS,
    build_annotation_queue,
    build_identity_groups,
    clean,
    export_ground_truth_rows,
    group_summary,
    load_csv,
    queue_fieldnames,
    select_identity_groups,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINAL_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "final_results"
    / "passport_extraction_results.csv"
)

GT_DIR = PROJECT_ROOT / "ground_truth"
MAIN_GT = GT_DIR / "passport_ground_truth.csv"

DEV2_QUEUE = GT_DIR / "annotation_queue_dev2.csv"
HOLDOUT2_QUEUE = GT_DIR / "annotation_queue_holdout2.csv"
SELECTION_CSV = GT_DIR / "gt_expansion_v7_selection.csv"


def normalized_text(value: Any) -> str:
    return " ".join(clean(value).upper().split())


def normalized_passport(value: Any) -> str:
    return "".join(
        ch
        for ch in normalized_text(value)
        if ch.isalnum()
    )


def normalized_name(value: Any) -> str:
    return "".join(
        ch
        for ch in normalized_text(value)
        if ch.isalpha()
    )


def normalized_date(value: Any) -> str:
    return clean(value)


def gt_identity_signatures(
    gt_rows: list[dict[str, str]],
) -> tuple[set[str], set[str], set[str]]:
    """
    Build conservative signatures from already verified GT.

    Exclusion uses:
    1) sample_id
    2) exact passport number
    3) exact surname + DOB + expiry

    This reduces the chance that an unannotated image variant of an
    existing real passport leaks into DEV2/HOLDOUT2.
    """
    sample_ids: set[str] = set()
    passport_numbers: set[str] = set()
    sde_signatures: set[str] = set()

    for row in gt_rows:
        sample_id = clean(row.get("sample_id"))
        if sample_id:
            sample_ids.add(sample_id)

        passport_number = normalized_passport(
            row.get("passport_number")
        )
        if passport_number:
            passport_numbers.add(passport_number)

        surname = normalized_name(
            row.get("surname")
        )
        dob = normalized_date(
            row.get("date_of_birth")
        )
        expiry = normalized_date(
            row.get("date_of_expiry")
        )

        if surname and dob and expiry:
            sde_signatures.add(
                f"{surname}|{dob}|{expiry}"
            )

    return (
        sample_ids,
        passport_numbers,
        sde_signatures,
    )


def member_matches_existing_gt(
    member: dict[str, str],
    *,
    gt_sample_ids: set[str],
    gt_passport_numbers: set[str],
    gt_sde_signatures: set[str],
) -> bool:
    sample_id = clean(
        member.get("sample_id")
    )

    if sample_id in gt_sample_ids:
        return True

    passport_number = normalized_passport(
        member.get("passport_number")
    )

    if (
        passport_number
        and passport_number
        in gt_passport_numbers
    ):
        return True

    surname = normalized_name(
        member.get("surname")
    )
    dob = normalized_date(
        member.get("date_of_birth")
    )
    expiry = normalized_date(
        member.get("date_of_expiry")
    )

    if surname and dob and expiry:
        signature = (
            f"{surname}|{dob}|{expiry}"
        )

        if signature in gt_sde_signatures:
            return True

    return False


def unseen_identity_groups(
    final_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
) -> tuple[
    dict[str, list[dict[str, str]]],
    int,
]:
    groups = build_identity_groups(
        final_rows
    )

    (
        gt_sample_ids,
        gt_passport_numbers,
        gt_sde_signatures,
    ) = gt_identity_signatures(
        gt_rows
    )

    eligible: dict[
        str,
        list[dict[str, str]]
    ] = {}

    excluded = 0

    for identity_id, members in groups.items():
        overlaps = any(
            member_matches_existing_gt(
                member,
                gt_sample_ids=gt_sample_ids,
                gt_passport_numbers=(
                    gt_passport_numbers
                ),
                gt_sde_signatures=(
                    gt_sde_signatures
                ),
            )
            for member in members
        )

        if overlaps:
            excluded += 1
            continue

        eligible[identity_id] = members

    return eligible, excluded


def stable_fraction(text: str) -> float:
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return (
        int(digest[:12], 16)
        / float(16 ** 12)
    )


def choose_holdout_ids(
    selected: list[dict[str, Any]],
    holdout_count: int,
) -> set[str]:
    """
    Choose HOLDOUT2 deterministically while preserving a rough
    difficulty/country spread.

    Within each (difficulty, country) stratum the stable hash is used,
    so rerunning on unchanged inputs returns the same split.
    """
    if holdout_count <= 0:
        return set()

    if holdout_count >= len(selected):
        return {
            clean(
                row.get(
                    "suggested_identity_id"
                )
            )
            for row in selected
        }

    strata: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in selected:
        key = (
            clean(
                row.get("difficulty")
            )
            or "unknown",
            clean(
                row.get("issuing_country")
            )
            or "UNKNOWN",
        )
        strata[key].append(row)

    # First allocate proportionally by stratum.
    total = len(selected)
    quotas: dict[
        tuple[str, str],
        int,
    ] = {}

    remainders: list[
        tuple[float, tuple[str, str]]
    ] = []

    allocated = 0

    for key, rows in strata.items():
        exact = (
            len(rows)
            * holdout_count
            / total
        )
        base = int(exact)
        quotas[key] = base
        allocated += base
        remainders.append(
            (exact - base, key)
        )

    for _, key in sorted(
        remainders,
        key=lambda item: (
            -item[0],
            item[1],
        ),
    ):
        if allocated >= holdout_count:
            break

        if quotas[key] < len(strata[key]):
            quotas[key] += 1
            allocated += 1

    holdout_ids: set[str] = set()

    for key, rows in strata.items():
        ranked = sorted(
            rows,
            key=lambda row: (
                stable_fraction(
                    clean(
                        row.get(
                            "suggested_identity_id"
                        )
                    )
                ),
                clean(
                    row.get(
                        "suggested_identity_id"
                    )
                ),
            ),
        )

        for row in ranked[
            : quotas.get(key, 0)
        ]:
            holdout_ids.add(
                clean(
                    row.get(
                        "suggested_identity_id"
                    )
                )
            )

    # Rare quota rounding fallback.
    if len(holdout_ids) < holdout_count:
        remaining = [
            row
            for row in selected
            if clean(
                row.get(
                    "suggested_identity_id"
                )
            )
            not in holdout_ids
        ]

        remaining.sort(
            key=lambda row: (
                stable_fraction(
                    clean(
                        row.get(
                            "suggested_identity_id"
                        )
                    )
                ),
                clean(
                    row.get(
                        "suggested_identity_id"
                    )
                ),
            )
        )

        for row in remaining:
            if (
                len(holdout_ids)
                >= holdout_count
            ):
                break

            holdout_ids.add(
                clean(
                    row.get(
                        "suggested_identity_id"
                    )
                )
            )

    return holdout_ids


def rows_for_identity_ids(
    groups: dict[
        str,
        list[dict[str, str]]
    ],
    identity_ids: set[str],
) -> list[dict[str, str]]:
    rows: list[
        dict[str, str]
    ] = []

    for identity_id in sorted(
        identity_ids
    ):
        rows.extend(
            groups[identity_id]
        )

    return rows


def build_queue_for_ids(
    groups: dict[
        str,
        list[dict[str, str]]
    ],
    identity_ids: set[str],
    split_name: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    source_rows = rows_for_identity_ids(
        groups,
        identity_ids,
    )

    queue_rows, summaries = (
        build_annotation_queue(
            rows=source_rows,
            target_identities=len(
                identity_ids
            ),
        )
    )

    queue_rows = [
        row
        for row in queue_rows
        if clean(
            row.get(
                "suggested_identity_id"
            )
        )
        in identity_ids
    ]

    for row in queue_rows:
        row["split"] = split_name

    summaries = [
        row
        for row in summaries
        if clean(
            row.get(
                "suggested_identity_id"
            )
        )
        in identity_ids
    ]

    return queue_rows, summaries


def prepare(
    dev2_identities: int,
    holdout2_identities: int,
    force: bool,
) -> None:
    if (
        dev2_identities <= 0
        or holdout2_identities <= 0
    ):
        raise ValueError(
            "DEV2 và HOLDOUT2 đều phải có "
            "ít nhất 1 identity."
        )

    outputs = (
        DEV2_QUEUE,
        HOLDOUT2_QUEUE,
        SELECTION_CSV,
    )

    existing_outputs = [
        path
        for path in outputs
        if path.exists()
    ]

    if existing_outputs and not force:
        joined = "\n".join(
            str(path)
            for path in existing_outputs
        )

        raise FileExistsError(
            "GT Expansion V7 output đã tồn tại:\n"
            f"{joined}\n\n"
            "Dùng --force chỉ khi muốn tạo "
            "lại selection."
        )

    final_rows = load_csv(
        FINAL_CSV
    )

    gt_rows = load_csv(
        MAIN_GT
    )

    groups, excluded_count = (
        unseen_identity_groups(
            final_rows,
            gt_rows,
        )
    )

    target = (
        dev2_identities
        + holdout2_identities
    )

    if len(groups) < target:
        raise RuntimeError(
            "Không đủ unseen identities. "
            f"Cần {target}, chỉ còn "
            f"{len(groups)}."
        )

    selected = select_identity_groups(
        groups=groups,
        target_identities=target,
    )

    if len(selected) != target:
        raise RuntimeError(
            "Identity selector không trả "
            f"đủ {target} identities."
        )

    selected_ids = {
        clean(
            row.get(
                "suggested_identity_id"
            )
        )
        for row in selected
    }

    holdout_ids = choose_holdout_ids(
        selected,
        holdout2_identities,
    )

    dev_ids = (
        selected_ids
        - holdout_ids
    )

    if (
        len(dev_ids)
        != dev2_identities
        or len(holdout_ids)
        != holdout2_identities
    ):
        raise RuntimeError(
            "Split allocation không đúng "
            "số identity yêu cầu."
        )

    dev_queue, dev_summaries = (
        build_queue_for_ids(
            groups,
            dev_ids,
            "dev2",
        )
    )

    holdout_queue, holdout_summaries = (
        build_queue_for_ids(
            groups,
            holdout_ids,
            "holdout2",
        )
    )

    write_csv(
        DEV2_QUEUE,
        dev_queue,
        queue_fieldnames(),
    )

    write_csv(
        HOLDOUT2_QUEUE,
        holdout_queue,
        queue_fieldnames(),
    )

    selection_rows: list[
        dict[str, Any]
    ] = []

    for split_name, summaries in (
        ("dev2", dev_summaries),
        (
            "holdout2",
            holdout_summaries,
        ),
    ):
        for row in summaries:
            item = dict(row)
            item["split"] = split_name
            selection_rows.append(item)

    selection_rows.sort(
        key=lambda row: (
            clean(row.get("split")),
            clean(
                row.get(
                    "difficulty"
                )
            ),
            clean(
                row.get(
                    "issuing_country"
                )
            ),
            clean(
                row.get(
                    "suggested_identity_id"
                )
            ),
        )
    )

    write_csv(
        SELECTION_CSV,
        selection_rows,
    )

    dev_difficulty = Counter(
        clean(
            row.get("difficulty")
        )
        for row in dev_summaries
    )

    holdout_difficulty = Counter(
        clean(
            row.get("difficulty")
        )
        for row in holdout_summaries
    )

    print("=" * 76)
    print("GT EXPANSION V7 — PREPARED")
    print("=" * 76)

    print(
        f"Final-result samples       : "
        f"{len(final_rows)}"
    )
    print(
        f"Existing GT samples        : "
        f"{len(gt_rows)}"
    )
    print(
        f"Existing GT identities     : "
        f"{len({clean(r.get('identity_id')) for r in gt_rows if clean(r.get('identity_id'))})}"
    )
    print(
        f"Candidate groups excluded  : "
        f"{excluded_count}"
    )
    print(
        f"Eligible unseen groups     : "
        f"{len(groups)}"
    )

    print()
    print(
        f"DEV2 identities            : "
        f"{len(dev_ids)}"
    )
    print(
        f"DEV2 image variants        : "
        f"{len(dev_queue)}"
    )
    print(
        "DEV2 difficulty           : "
        f"{dict(dev_difficulty)}"
    )

    print()
    print(
        f"HOLDOUT2 identities        : "
        f"{len(holdout_ids)}"
    )
    print(
        f"HOLDOUT2 image variants    : "
        f"{len(holdout_queue)}"
    )
    print(
        "HOLDOUT2 difficulty       : "
        f"{dict(holdout_difficulty)}"
    )

    print()
    print(
        f"DEV2 queue    : {DEV2_QUEUE}"
    )
    print(
        f"HOLDOUT2 queue: {HOLDOUT2_QUEUE}"
    )
    print(
        f"Selection     : {SELECTION_CSV}"
    )

    print()
    print(
        "NEXT — annotate DEV2 only:"
    )
    print(
        "python src/annotation_assistant.py "
        "gui --queue "
        "ground_truth/annotation_queue_dev2.csv"
    )

    print()
    print(
        "Do NOT open HOLDOUT2 yet. "
        "Keep it sealed until the next "
        "pipeline version is frozen."
    )


def queue_status(
    queue_path: Path,
) -> tuple[int, int, int]:
    rows = load_csv(
        queue_path
    )

    anchors = [
        row
        for row in rows
        if clean(
            row.get("is_anchor")
        ).lower()
        in {"1", "true", "yes", "y"}
    ]

    verified = [
        row
        for row in anchors
        if clean(
            row.get(
                "annotation_status"
            )
        )
        == "verified"
    ]

    needs_review = [
        row
        for row in anchors
        if clean(
            row.get(
                "annotation_status"
            )
        )
        == "needs_review"
    ]

    return (
        len(anchors),
        len(verified),
        len(needs_review),
    )


def print_status() -> None:
    print("=" * 76)
    print("GT EXPANSION V7 — STATUS")
    print("=" * 76)

    for name, path in (
        ("DEV2", DEV2_QUEUE),
        ("HOLDOUT2", HOLDOUT2_QUEUE),
    ):
        if not path.exists():
            print(
                f"{name:<10}: NOT PREPARED"
            )
            continue

        total, verified, needs_review = (
            queue_status(path)
        )

        print(
            f"{name:<10}: "
            f"verified {verified}/{total}, "
            f"needs_review={needs_review}"
        )


def backup_path(
    original: Path,
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return original.with_name(
        original.stem
        + f".before_v7_{timestamp}"
        + original.suffix
    )


def merge_verified_queue(
    queue_path: Path,
    split_name: str,
    output_path: Path,
) -> None:
    if split_name not in {
        "dev2",
        "holdout2",
    }:
        raise ValueError(
            "split phải là dev2 hoặc "
            "holdout2."
        )

    existing = load_csv(
        output_path
    )

    queue_rows = load_csv(
        queue_path
    )

    exported = (
        export_ground_truth_rows(
            queue_rows
        )
    )

    if not exported:
        raise RuntimeError(
            "Queue chưa có identity "
            "verified để merge."
        )

    for row in exported:
        row["split"] = split_name

    existing_ids = {
        clean(
            row.get("sample_id")
        )
        for row in existing
    }

    duplicate_ids = sorted(
        {
            clean(
                row.get("sample_id")
            )
            for row in exported
            if clean(
                row.get("sample_id")
            )
            in existing_ids
        }
    )

    if duplicate_ids:
        raise RuntimeError(
            "Refusing merge: sample_id "
            "đã tồn tại trong GT:\n"
            + "\n".join(
                duplicate_ids[:20]
            )
        )

    # Prevent identity split leakage.
    existing_identity_split: dict[
        str,
        set[str]
    ] = defaultdict(set)

    for row in existing:
        identity_id = clean(
            row.get("identity_id")
        )
        split = clean(
            row.get("split")
        )

        if identity_id:
            existing_identity_split[
                identity_id
            ].add(split)

    leaked: list[str] = []

    for row in exported:
        identity_id = clean(
            row.get("identity_id")
        )

        if (
            identity_id
            and identity_id
            in existing_identity_split
            and split_name
            not in existing_identity_split[
                identity_id
            ]
        ):
            leaked.append(identity_id)

    if leaked:
        raise RuntimeError(
            "Identity split leakage detected:\n"
            + "\n".join(
                sorted(set(leaked))
            )
        )

    merged = [
        *existing,
        *exported,
    ]

    fieldnames = [
        "sample_id",
        "identity_id",
        "split",
        "filename",
        "relative_path",
        *FIELDS,
        "notes",
    ]

    backup = backup_path(
        output_path
    )

    shutil.copy2(
        output_path,
        backup,
    )

    write_csv(
        output_path,
        merged,
        fieldnames,
    )

    print("=" * 76)
    print("GT EXPANSION V7 — MERGED")
    print("=" * 76)
    print(
        f"Split             : "
        f"{split_name}"
    )
    print(
        f"New samples       : "
        f"{len(exported)}"
    )
    print(
        f"New identities    : "
        f"{len({clean(r.get('identity_id')) for r in exported})}"
    )
    print(
        f"Total GT samples  : "
        f"{len(merged)}"
    )
    print(
        f"Total identities  : "
        f"{len({clean(r.get('identity_id')) for r in merged if clean(r.get('identity_id'))})}"
    )
    print(
        f"Backup            : "
        f"{backup}"
    )
    print(
        f"Updated GT        : "
        f"{output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create unseen DEV2/HOLDOUT2 "
            "ground-truth expansion queues."
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    p_prepare = sub.add_parser(
        "prepare"
    )
    p_prepare.add_argument(
        "--dev2-identities",
        type=int,
        default=20,
    )
    p_prepare.add_argument(
        "--holdout2-identities",
        type=int,
        default=10,
    )
    p_prepare.add_argument(
        "--force",
        action="store_true",
    )

    sub.add_parser(
        "status"
    )

    p_merge = sub.add_parser(
        "merge"
    )
    p_merge.add_argument(
        "--queue",
        type=Path,
        required=True,
    )
    p_merge.add_argument(
        "--split",
        choices=(
            "dev2",
            "holdout2",
        ),
        required=True,
    )
    p_merge.add_argument(
        "--output",
        type=Path,
        default=MAIN_GT,
    )

    args = parser.parse_args()

    if args.command == "prepare":
        prepare(
            dev2_identities=(
                args.dev2_identities
            ),
            holdout2_identities=(
                args.holdout2_identities
            ),
            force=args.force,
        )

    elif args.command == "status":
        print_status()

    elif args.command == "merge":
        merge_verified_queue(
            queue_path=args.queue,
            split_name=args.split,
            output_path=args.output,
        )

    else:
        raise AssertionError(
            args.command
        )


if __name__ == "__main__":
    main()
