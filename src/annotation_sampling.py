from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


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

HARD_RISK_THRESHOLD = 6.0
MEDIUM_RISK_THRESHOLD = 2.5


@dataclass(frozen=True)
class IdentitySuggestion:
    suggested_identity_id: str
    confidence: str
    evidence: str
    signature: str


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file:\n{path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError("Không có row để ghi.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null"}:
        return ""
    return text


def compact_alnum(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", clean(value).upper())


def compact_alpha(value: Any) -> str:
    return re.sub(r"[^A-Z]", "", clean(value).upper())


def normalize_date(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        # Existing pipeline normally emits DD/MM/YYYY. Keep order stable.
        return digits
    return text.upper()


def normalized_field(field: str, value: Any) -> str:
    if field in {"passport_number"}:
        return compact_alnum(value)
    if field in {"surname", "given_names", "nationality", "sex"}:
        return compact_alpha(value)
    if field in {"date_of_birth", "date_of_expiry", "date_of_issue"}:
        return normalize_date(value)
    return clean(value).upper()


def stable_short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y"}


def falsey(value: Any) -> bool:
    return clean(value).lower() in {"0", "false", "no", "n"}


def quality_good(value: Any) -> bool:
    return clean(value).lower() in {
        "verified",
        "strong",
        "high",
        "high_confidence",
        "medium",
        "medium_confidence",
    }


def quality_strong(value: Any) -> bool:
    return clean(value).lower() in {
        "verified",
        "strong",
        "high",
        "high_confidence",
    }


def source_strong(value: Any) -> bool:
    text = clean(value).lower()
    return text.startswith("mrz_verified") or text.startswith("mrz_strong") or text.startswith("viz_high")


def suggest_identity(row: dict[str, str]) -> IdentitySuggestion:
    sample_id = clean(row.get("sample_id")) or stable_short_hash(clean(row.get("relative_path")))

    passport_number = normalized_field("passport_number", row.get("passport_number"))
    passport_quality = clean(row.get("passport_number_quality"))
    passport_source = clean(row.get("passport_number_source"))

    surname = normalized_field("surname", row.get("surname"))
    dob = normalized_field("date_of_birth", row.get("date_of_birth"))
    expiry = normalized_field("date_of_expiry", row.get("date_of_expiry"))

    surname_q = clean(row.get("surname_quality"))
    dob_q = clean(row.get("date_of_birth_quality"))
    expiry_q = clean(row.get("date_of_expiry_quality"))

    # Highest-confidence grouping: exact passport number backed by a strong source.
    if passport_number and (quality_strong(passport_quality) or source_strong(passport_source)):
        signature = f"PPN:{passport_number}"
        return IdentitySuggestion(
            suggested_identity_id=f"id_ppn_{stable_short_hash(signature)}",
            confidence="high",
            evidence="strong_passport_number",
            signature=signature,
        )

    # Conservative secondary grouping. Requiring three stable document fields makes
    # accidental merges much less likely than fuzzy name-only grouping.
    if surname and dob and expiry and all(
        quality_good(value) for value in (surname_q, dob_q, expiry_q)
    ):
        signature = f"SDE:{surname}|{dob}|{expiry}"
        return IdentitySuggestion(
            suggested_identity_id=f"id_sde_{stable_short_hash(signature)}",
            confidence="medium",
            evidence="surname_dob_expiry",
            signature=signature,
        )

    # Weak passport numbers are intentionally NOT used alone for automatic grouping.
    # They are useful as a near-duplicate hint, but a bad OCR character must not merge
    # unrelated passports into the same identity/split.
    signature = f"SINGLE:{sample_id}"
    return IdentitySuggestion(
        suggested_identity_id=f"id_single_{stable_short_hash(signature)}",
        confidence="single",
        evidence="singleton_safe_fallback",
        signature=signature,
    )


def risk_score(row: dict[str, str]) -> float:
    score = 0.0

    quality_status = clean(row.get("quality_status")).lower()
    score += {
        "low_confidence": 5.0,
        "review": 4.0,
        "medium_confidence": 1.8,
        "high_confidence": 0.0,
    }.get(quality_status, 1.0)

    coverage = clean(row.get("coverage_status") or row.get("final_status")).lower()
    score += {
        "failed": 5.0,
        "partial": 3.0,
        "complete": 0.0,
    }.get(coverage, 1.0)

    if truthy(row.get("review_required")):
        score += 2.0

    conflicts = [part for part in clean(row.get("source_conflict_fields")).split("|") if part.strip()]
    score += min(4.0, len(conflicts) * 1.5)

    try:
        score += 2.0 * int(float(clean(row.get("consistency_high_count")) or 0))
        score += 0.8 * int(float(clean(row.get("consistency_medium_count")) or 0))
    except ValueError:
        pass

    for key in (
        "passport_number_check_valid",
        "birth_date_check_valid",
        "expiry_date_check_valid",
        "final_check_valid",
    ):
        if falsey(row.get(key)):
            score += 1.0

    if clean(row.get("mrz_parse_mode")).lower() == "padded_or_truncated":
        score += 1.5

    doi_status = clean(row.get("doi_status")).lower()
    if not clean(row.get("date_of_issue")):
        score += 1.5
    elif doi_status in {"low_confidence", "weak"}:
        score += 1.0

    missing = [part for part in clean(row.get("missing_fields")).split("|") if part.strip()]
    score += min(4.0, 0.75 * len(missing))

    return round(score, 3)


def difficulty_from_risk(score: float) -> str:
    if score >= HARD_RISK_THRESHOLD:
        return "hard"
    if score >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "easy"


def annotation_ease_score(row: dict[str, str]) -> float:
    """Higher is better for choosing one visually convenient anchor per identity."""
    score = 0.0
    coverage = clean(row.get("coverage_status")).lower()
    quality = clean(row.get("quality_status")).lower()
    score += {"complete": 5.0, "partial": 2.0, "failed": 0.0}.get(coverage, 0.0)
    score += {
        "high_confidence": 4.0,
        "medium_confidence": 2.0,
        "review": 1.0,
        "low_confidence": 0.0,
    }.get(quality, 0.0)
    score += sum(bool(clean(row.get(field))) for field in FIELDS) * 0.5
    score -= risk_score(row) * 0.15
    return round(score, 3)


def build_identity_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        suggestion = suggest_identity(row)
        enriched = dict(row)
        enriched["suggested_identity_id"] = suggestion.suggested_identity_id
        enriched["identity_confidence"] = suggestion.confidence
        enriched["identity_evidence"] = suggestion.evidence
        enriched["identity_signature"] = suggestion.signature
        enriched["annotation_risk_score"] = str(risk_score(row))
        enriched["annotation_difficulty"] = difficulty_from_risk(risk_score(row))
        groups[suggestion.suggested_identity_id].append(enriched)
    return dict(groups)


def group_summary(identity_id: str, members: list[dict[str, str]]) -> dict[str, Any]:
    risks = [risk_score(member) for member in members]
    max_risk = max(risks) if risks else 0.0
    mean_risk = sum(risks) / len(risks) if risks else 0.0
    countries = [clean(member.get("issuing_country")) or "UNKNOWN" for member in members]
    country = max(set(countries), key=countries.count) if countries else "UNKNOWN"
    anchor = max(
        members,
        key=lambda row: (annotation_ease_score(row), clean(row.get("sample_id"))),
    )
    return {
        "suggested_identity_id": identity_id,
        "identity_confidence": clean(anchor.get("identity_confidence")),
        "identity_evidence": clean(anchor.get("identity_evidence")),
        "group_size": len(members),
        "max_risk": round(max_risk, 3),
        "mean_risk": round(mean_risk, 3),
        "difficulty": difficulty_from_risk(max_risk),
        "issuing_country": country,
        "anchor_sample_id": clean(anchor.get("sample_id")),
    }


def _round_robin_by_country(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for group in sorted(
        groups,
        key=lambda item: (-float(item["max_risk"]), item["suggested_identity_id"]),
    ):
        buckets[clean(group.get("issuing_country")) or "UNKNOWN"].append(group)

    countries = sorted(
        buckets,
        key=lambda country: (-len(buckets[country]), country),
    )
    output: list[dict[str, Any]] = []
    while any(buckets[country] for country in countries):
        for country in countries:
            if buckets[country]:
                output.append(buckets[country].popleft())
    return output


def select_identity_groups(
    groups: dict[str, list[dict[str, str]]],
    target_identities: int,
) -> list[dict[str, Any]]:
    if target_identities <= 0:
        raise ValueError("target_identities phải > 0")

    summaries = [group_summary(identity_id, members) for identity_id, members in groups.items()]
    target = min(target_identities, len(summaries))

    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_difficulty[summary["difficulty"]].append(summary)
    for difficulty in by_difficulty:
        by_difficulty[difficulty] = _round_robin_by_country(by_difficulty[difficulty])

    # Deliberately overweight hard cases while preserving a calibration slice of easy cases.
    desired = {
        "hard": math.ceil(target * 0.50),
        "medium": math.floor(target * 0.30),
    }
    desired["easy"] = max(0, target - desired["hard"] - desired["medium"])

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for difficulty in ("hard", "medium", "easy"):
        for summary in by_difficulty.get(difficulty, [])[: desired[difficulty]]:
            selected.append(summary)
            selected_ids.add(summary["suggested_identity_id"])

    # Fill any shortage from the best remaining identities, still country-diversified.
    remaining = [summary for summary in summaries if summary["suggested_identity_id"] not in selected_ids]
    remaining = _round_robin_by_country(remaining)
    remaining.sort(key=lambda item: (-float(item["max_risk"]), item["issuing_country"], item["suggested_identity_id"]))
    for summary in remaining:
        if len(selected) >= target:
            break
        selected.append(summary)
        selected_ids.add(summary["suggested_identity_id"])

    return selected[:target]


def build_annotation_queue(
    rows: list[dict[str, str]],
    target_identities: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = build_identity_groups(rows)
    selected_summaries = select_identity_groups(groups, target_identities)
    selected_ids = {item["suggested_identity_id"] for item in selected_summaries}
    summary_by_id = {item["suggested_identity_id"]: item for item in selected_summaries}

    queue_rows: list[dict[str, Any]] = []
    for identity_id in selected_ids:
        members = groups[identity_id]
        summary = summary_by_id[identity_id]
        anchor_id = summary["anchor_sample_id"]
        for member in members:
            sample_id = clean(member.get("sample_id"))
            item: dict[str, Any] = {
                "sample_id": sample_id,
                "suggested_identity_id": identity_id,
                "identity_id": identity_id,
                "identity_confidence": member["identity_confidence"],
                "identity_evidence": member["identity_evidence"],
                "group_size": len(members),
                "is_anchor": sample_id == anchor_id,
                "annotation_status": "pending" if sample_id == anchor_id else "waiting_for_anchor",
                "propagated_from_sample_id": "",
                "filename": clean(member.get("filename")),
                "relative_path": clean(member.get("relative_path")),
                "generated_filename": clean(member.get("generated_filename")) or f"{sample_id}.jpg",
                "split": "",
                "issuing_country": clean(member.get("issuing_country")),
                "coverage_status": clean(member.get("coverage_status")),
                "quality_status": clean(member.get("quality_status")),
                "quality_score": clean(member.get("quality_score")),
                "review_required": clean(member.get("review_required")),
                "review_reasons": clean(member.get("review_reasons")),
                "source_conflict_fields": clean(member.get("source_conflict_fields")),
                "consistency_issue_codes": clean(member.get("consistency_issue_codes")),
                "annotation_risk_score": member["annotation_risk_score"],
                "annotation_difficulty": member["annotation_difficulty"],
                "notes": "",
            }
            for field in FIELDS:
                prediction = clean(member.get(field))
                item[f"pred_{field}"] = prediction
                item[f"gt_{field}"] = prediction
                item[f"{field}_source"] = clean(member.get(f"{field}_source"))
                item[f"{field}_quality"] = clean(member.get(f"{field}_quality"))
            queue_rows.append(item)

    # Anchors first, hard identities first; non-anchors remain nearby by identity.
    difficulty_order = {"hard": 0, "medium": 1, "easy": 2}
    queue_rows.sort(
        key=lambda row: (
            difficulty_order.get(clean(row.get("annotation_difficulty")), 9),
            -float(clean(row.get("annotation_risk_score")) or 0.0),
            clean(row.get("suggested_identity_id")),
            0 if truthy(row.get("is_anchor")) else 1,
            clean(row.get("sample_id")),
        )
    )
    selected_summaries.sort(
        key=lambda item: (
            difficulty_order.get(item["difficulty"], 9),
            -float(item["max_risk"]),
            item["suggested_identity_id"],
        )
    )
    return queue_rows, selected_summaries


def _same_identity_hint_score(left: dict[str, str], right: dict[str, str]) -> tuple[float, str]:
    if clean(left.get("suggested_identity_id")) == clean(right.get("suggested_identity_id")):
        return 0.0, "already_grouped"

    l_ppn = normalized_field("passport_number", left.get("passport_number"))
    r_ppn = normalized_field("passport_number", right.get("passport_number"))
    l_dob = normalized_field("date_of_birth", left.get("date_of_birth"))
    r_dob = normalized_field("date_of_birth", right.get("date_of_birth"))
    l_exp = normalized_field("date_of_expiry", left.get("date_of_expiry"))
    r_exp = normalized_field("date_of_expiry", right.get("date_of_expiry"))
    l_sur = normalized_field("surname", left.get("surname"))
    r_sur = normalized_field("surname", right.get("surname"))

    evidence: list[str] = []
    score = 0.0

    if l_dob and l_dob == r_dob:
        score += 3.0
        evidence.append("same_dob")
    if l_exp and l_exp == r_exp:
        score += 2.5
        evidence.append("same_expiry")
    if l_sur and r_sur:
        ratio = SequenceMatcher(a=l_sur, b=r_sur).ratio()
        if ratio >= 0.90:
            score += 2.5
            evidence.append(f"surname_sim={ratio:.2f}")
    if l_ppn and r_ppn:
        ratio = SequenceMatcher(a=l_ppn, b=r_ppn).ratio()
        if ratio >= 0.85:
            score += 3.0
            evidence.append(f"passport_sim={ratio:.2f}")

    return score, ",".join(evidence)


def near_duplicate_identity_suggestions(
    rows: list[dict[str, str]],
    max_pairs: int = 200,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        suggestion = suggest_identity(row)
        item["suggested_identity_id"] = suggestion.suggested_identity_id
        enriched.append(item)

    candidates: list[dict[str, Any]] = []
    # O(n^2) is acceptable for the ~1k-image datasets this repository targets.
    for i in range(len(enriched)):
        left = enriched[i]
        for j in range(i + 1, len(enriched)):
            right = enriched[j]
            score, evidence = _same_identity_hint_score(left, right)
            if score < 6.0:
                continue
            candidates.append(
                {
                    "left_sample_id": clean(left.get("sample_id")),
                    "right_sample_id": clean(right.get("sample_id")),
                    "left_identity": clean(left.get("suggested_identity_id")),
                    "right_identity": clean(right.get("suggested_identity_id")),
                    "hint_score": round(score, 3),
                    "evidence": evidence,
                    "left_passport_number": clean(left.get("passport_number")),
                    "right_passport_number": clean(right.get("passport_number")),
                    "left_surname": clean(left.get("surname")),
                    "right_surname": clean(right.get("surname")),
                    "left_date_of_birth": clean(left.get("date_of_birth")),
                    "right_date_of_birth": clean(right.get("date_of_birth")),
                }
            )
    candidates.sort(key=lambda item: (-float(item["hint_score"]), item["left_sample_id"], item["right_sample_id"]))
    return candidates[:max_pairs]


def queue_fieldnames() -> list[str]:
    base = [
        "sample_id",
        "suggested_identity_id",
        "identity_id",
        "identity_confidence",
        "identity_evidence",
        "group_size",
        "is_anchor",
        "annotation_status",
        "propagated_from_sample_id",
        "filename",
        "relative_path",
        "generated_filename",
        "split",
        "issuing_country",
        "coverage_status",
        "quality_status",
        "quality_score",
        "review_required",
        "review_reasons",
        "source_conflict_fields",
        "consistency_issue_codes",
        "annotation_risk_score",
        "annotation_difficulty",
        "notes",
    ]
    for field in FIELDS:
        base.extend([f"pred_{field}", f"gt_{field}", f"{field}_source", f"{field}_quality"])
    return base


def propagate_anchor(
    rows: list[dict[str, str]],
    anchor_sample_id: str,
    identity_id: str,
    gt_values: dict[str, str],
    notes: str = "",
    propagate: bool = True,
) -> int:
    anchor: dict[str, str] | None = None
    for row in rows:
        if clean(row.get("sample_id")) == anchor_sample_id:
            anchor = row
            break
    if anchor is None:
        raise KeyError(f"Không tìm thấy anchor sample_id={anchor_sample_id}")

    group_id = clean(anchor.get("suggested_identity_id"))
    anchor["identity_id"] = identity_id
    anchor["annotation_status"] = "verified"
    anchor["notes"] = notes
    for field in FIELDS:
        anchor[f"gt_{field}"] = clean(gt_values.get(field))

    affected = 1
    if not propagate:
        return affected

    for row in rows:
        if row is anchor:
            continue
        if clean(row.get("suggested_identity_id")) != group_id:
            continue
        row["identity_id"] = identity_id
        row["annotation_status"] = "propagated"
        row["propagated_from_sample_id"] = anchor_sample_id
        for field in FIELDS:
            row[f"gt_{field}"] = clean(gt_values.get(field))
        affected += 1
    return affected


def mark_needs_review(rows: list[dict[str, str]], sample_id: str, notes: str = "") -> None:
    for row in rows:
        if clean(row.get("sample_id")) == sample_id:
            row["annotation_status"] = "needs_review"
            row["notes"] = notes
            return
    raise KeyError(f"Không tìm thấy sample_id={sample_id}")


def export_ground_truth_rows(queue_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    allowed = {"verified", "propagated"}
    for row in queue_rows:
        if clean(row.get("annotation_status")) not in allowed:
            continue
        identity_id = clean(row.get("identity_id"))
        if not identity_id:
            continue
        item = {
            "sample_id": clean(row.get("sample_id")),
            "identity_id": identity_id,
            "split": clean(row.get("split")),
            "filename": clean(row.get("filename")),
            "relative_path": clean(row.get("relative_path")),
        }
        for field in FIELDS:
            item[field] = clean(row.get(f"gt_{field}"))
        item["notes"] = clean(row.get("notes"))
        output.append(item)
    return output


def progress_summary(queue_rows: list[dict[str, str]]) -> dict[str, Any]:
    anchors = [row for row in queue_rows if truthy(row.get("is_anchor"))]
    verified_anchors = [row for row in anchors if clean(row.get("annotation_status")) == "verified"]
    covered = [row for row in queue_rows if clean(row.get("annotation_status")) in {"verified", "propagated"}]
    needs_review = [row for row in anchors if clean(row.get("annotation_status")) == "needs_review"]
    pending = [row for row in anchors if clean(row.get("annotation_status")) in {"pending", ""}]
    manual_saving_factor = (len(covered) / len(verified_anchors)) if verified_anchors else 0.0
    return {
        "selected_identities": len(anchors),
        "verified_identities": len(verified_anchors),
        "pending_identities": len(pending),
        "needs_review_identities": len(needs_review),
        "covered_samples": len(covered),
        "total_selected_samples": len(queue_rows),
        "manual_saving_factor": round(manual_saving_factor, 2),
    }
