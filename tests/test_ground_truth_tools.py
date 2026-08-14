from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "ground_truth_tools.py"
spec = importlib.util.spec_from_file_location("ground_truth_tools", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class GroundTruthToolsTests(unittest.TestCase):
    def test_identity_order_is_deterministic(self) -> None:
        ids = ["p3", "p1", "p2", "p4"]
        self.assertEqual(
            module.deterministic_identity_order(ids, 42),
            module.deterministic_identity_order(list(reversed(ids)), 42),
        )

    def test_different_seed_can_change_order(self) -> None:
        ids = [f"p{i}" for i in range(20)]
        self.assertNotEqual(
            module.deterministic_identity_order(ids, 1),
            module.deterministic_identity_order(ids, 2),
        )


if __name__ == "__main__":
    unittest.main()
