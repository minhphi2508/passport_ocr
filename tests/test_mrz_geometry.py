from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from mrz_geometry import reconstruct_mrz_lines


class MRZGeometryTests(unittest.TestCase):
    def test_geometry_reconstructs_two_rows_left_to_right(self) -> None:
        items = [
            {"text": "<<<<<<<<<<<<<<<<<<<<<<", "box": [220, 70, 440, 95]},
            {"text": "P<UTOERIKSSON<<ANNA<MARIA", "box": [0, 15, 260, 40]},
            {"text": "<<<<<<<<<<<<<<<<<<", "box": [260, 15, 440, 40]},
            {"text": "L898902C36UTO7408122F1204159", "box": [0, 70, 290, 95]},
        ]

        lines, method = reconstruct_mrz_lines(items)

        self.assertEqual(method, "geometry_two_rows")
        self.assertTrue(lines[0].startswith("P<UTO"))
        self.assertTrue(lines[1].startswith("L898902C36"))

    def test_same_row_fragments_do_not_force_false_two_rows(self) -> None:
        items = [
            {"text": "P<UTOERIKSSON", "box": [0, 15, 150, 40]},
            {"text": "<<ANNA<MARIA", "box": [160, 17, 300, 42]},
        ]

        lines, method = reconstruct_mrz_lines(items)
        self.assertEqual(method, "text_order_fallback")
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
