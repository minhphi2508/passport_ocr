from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from annotation_sampling import (  # noqa: E402
    FIELDS,
    build_annotation_queue,
    build_identity_groups,
    export_ground_truth_rows,
    progress_summary,
    propagate_anchor,
    suggest_identity,
)


def base_row(sample_id: str, ppn: str = "A1234567") -> dict[str, str]:
    row = {
        "sample_id": sample_id,
        "filename": f"{sample_id}.jpg",
        "relative_path": f"set/{sample_id}.jpg",
        "generated_filename": f"{sample_id}.jpg",
        "passport_number": ppn,
        "surname": "NGUYEN",
        "given_names": "AN",
        "nationality": "VNM",
        "date_of_birth": "01/01/1990",
        "sex": "M",
        "date_of_expiry": "01/01/2030",
        "date_of_issue": "01/01/2020",
        "passport_number_quality": "verified",
        "passport_number_source": "mrz_verified",
        "surname_quality": "strong",
        "date_of_birth_quality": "verified",
        "date_of_expiry_quality": "verified",
        "coverage_status": "complete",
        "quality_status": "high_confidence",
        "review_required": "False",
        "issuing_country": "VNM",
    }
    for field in FIELDS:
        row.setdefault(f"{field}_quality", "high")
        row.setdefault(f"{field}_source", "mrz_strong")
    return row


class AnnotationSamplingTests(unittest.TestCase):
    def test_strong_passport_number_groups_variants(self) -> None:
        left = base_row("s1")
        right = base_row("s2")
        groups = build_identity_groups([left, right])
        self.assertEqual(len(groups), 1)
        members = next(iter(groups.values()))
        self.assertEqual(len(members), 2)
        self.assertEqual(members[0]["identity_confidence"], "high")

    def test_weak_passport_number_alone_does_not_group(self) -> None:
        left = base_row("s1")
        right = base_row("s2")
        for row in (left, right):
            row["passport_number_quality"] = "weak"
            row["passport_number_source"] = "mrz_weak_fallback"
            row["surname_quality"] = "weak"
            row["date_of_birth_quality"] = "weak"
            row["date_of_expiry_quality"] = "weak"
        self.assertNotEqual(
            suggest_identity(left).suggested_identity_id,
            suggest_identity(right).suggested_identity_id,
        )

    def test_anchor_annotation_propagates_to_group(self) -> None:
        rows = [base_row("s1"), base_row("s2")]
        queue, _ = build_annotation_queue(rows, target_identities=1)
        anchor = next(row for row in queue if str(row["is_anchor"]).lower() == "true")
        values = {field: f"GT_{field}" for field in FIELDS}
        affected = propagate_anchor(
            queue,
            anchor_sample_id=str(anchor["sample_id"]),
            identity_id="passport_001",
            gt_values=values,
            propagate=True,
        )
        self.assertEqual(affected, 2)
        self.assertEqual({row["identity_id"] for row in queue}, {"passport_001"})
        self.assertEqual(
            {row["annotation_status"] for row in queue},
            {"verified", "propagated"},
        )
        self.assertTrue(all(row["gt_surname"] == "GT_surname" for row in queue))

    def test_export_only_verified_or_propagated(self) -> None:
        rows = [base_row("s1"), base_row("s2"), base_row("s3", ppn="B7654321")]
        queue, _ = build_annotation_queue(rows, target_identities=2)
        anchor = next(row for row in queue if row["suggested_identity_id"].startswith("id_ppn_") and str(row["is_anchor"]).lower() == "true")
        values = {field: str(anchor[f"pred_{field}"]) for field in FIELDS}
        propagate_anchor(
            queue,
            anchor_sample_id=str(anchor["sample_id"]),
            identity_id="passport_001",
            gt_values=values,
            propagate=True,
        )
        exported = export_ground_truth_rows(queue)
        self.assertGreaterEqual(len(exported), 1)
        self.assertTrue(all(row["identity_id"] == "passport_001" for row in exported))

    def test_progress_reports_manual_saving(self) -> None:
        queue, _ = build_annotation_queue([base_row("s1"), base_row("s2")], target_identities=1)
        anchor = next(row for row in queue if str(row["is_anchor"]).lower() == "true")
        values = {field: str(anchor[f"pred_{field}"]) for field in FIELDS}
        propagate_anchor(
            queue,
            anchor_sample_id=str(anchor["sample_id"]),
            identity_id="passport_001",
            gt_values=values,
            propagate=True,
        )
        summary = progress_summary(queue)
        self.assertEqual(summary["verified_identities"], 1)
        self.assertEqual(summary["covered_samples"], 2)
        self.assertEqual(summary["manual_saving_factor"], 2.0)


if __name__ == "__main__":
    unittest.main()
