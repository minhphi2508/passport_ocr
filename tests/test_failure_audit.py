from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "failure_audit.py"
spec = importlib.util.spec_from_file_location("failure_audit", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class FailureAuditTests(unittest.TestCase):
    def test_fusion_when_alternative_viz_is_correct(self) -> None:
        category, _ = module.classify_error(
            field="passport_number",
            expected="A1234567",
            predicted="A1234568",
            final_row={"passport_number_source": "mrz_strong", "passport_stage_status": "success"},
            processing_row={},
            mrz_row={"passport_number": "A1234568", "parse_status": "success", "parse_mode": "strict_44_44"},
            viz_row={"passport_number": "A1234567"},
            doi_row={},
        )
        self.assertEqual(category, "fusion")

    def test_doi_missing(self) -> None:
        category, _ = module.classify_error(
            field="date_of_issue",
            expected="01/01/2020",
            predicted=None,
            final_row={"passport_stage_status": "success"},
            processing_row={},
            mrz_row={},
            viz_row={},
            doi_row={},
        )
        self.assertEqual(category, "doi_extraction")

    def test_passport_stage_has_priority(self) -> None:
        category, _ = module.classify_error(
            field="surname",
            expected="DOE",
            predicted=None,
            final_row={"passport_stage_status": "no_passport_evidence"},
            processing_row={}, mrz_row={}, viz_row={}, doi_row={},
        )
        self.assertEqual(category, "passport_stage")


if __name__ == "__main__":
    unittest.main()
