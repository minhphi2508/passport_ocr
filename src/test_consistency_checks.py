from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from consistency_checks import analyze_final_record


class ConsistencyTests(unittest.TestCase):
    def test_high_conflict_between_verified_mrz_and_high_viz(self) -> None:
        final_fields = {
            "passport_number": "A1234567",
            "surname": "DOE",
            "given_names": "JOHN",
            "nationality": "USA",
            "date_of_birth": "1990-01-01",
            "sex": "M",
            "date_of_expiry": "2030-01-01",
            "date_of_issue": "2020-01-01",
        }
        mrz = {
            "passport_number": "A1234567",
            "passport_number_check_valid": "True",
            "birth_date": "1990-01-01",
            "birth_date_check_valid": "True",
            "expiry_date": "2030-01-01",
            "expiry_date_check_valid": "True",
            "parse_mode": "strict_44_44",
            "all_main_checks_valid": "True",
            "surname": "DOE",
        }
        viz = {
            "passport_number": "B7654321",
            "passport_number_score": "9.2",
            "passport_number_variant_agreement": "2",
        }
        qualities = {
            "passport_number": "verified",
            "surname": "strong",
            "given_names": "strong",
            "nationality": "strong",
            "date_of_birth": "verified",
            "sex": "strong",
            "date_of_expiry": "verified",
            "date_of_issue": "high_confidence",
        }

        result = analyze_final_record(final_fields, qualities, mrz, viz)
        self.assertIn("passport_number", result["source_conflict_fields"])
        self.assertGreaterEqual(result["consistency_high_count"], 1)

    def test_temporal_error_is_flagged(self) -> None:
        final_fields = {
            "date_of_birth": "1990-01-01",
            "date_of_issue": "2030-01-01",
            "date_of_expiry": "2025-01-01",
        }
        result = analyze_final_record(
            final_fields,
            {"date_of_birth": "verified", "date_of_issue": "medium_confidence", "date_of_expiry": "verified"},
            {},
            {},
        )
        codes = {item["code"] for item in result["consistency_issues"]}
        self.assertIn("expiry_not_after_issue", codes)


if __name__ == "__main__":
    unittest.main()
