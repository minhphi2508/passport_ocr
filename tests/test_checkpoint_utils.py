from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from checkpoint_utils import append_jsonl, load_jsonl, prepare_stage_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_jsonl_survives_truncated_last_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.jsonl"
            append_jsonl(path, {"filename": "a.jpg"}, durable=False)
            with path.open("a", encoding="utf-8") as file:
                file.write('{"filename":')
            records = load_jsonl(path)
            self.assertEqual(records, [{"filename": "a.jpg"}])

    def test_changed_fingerprint_invalidates_stage_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            checkpoint = output / "checkpoint.jsonl"
            checkpoint.write_text("{}\n", encoding="utf-8")

            prepare_stage_checkpoint(
                output,
                "test",
                "fp1",
                {"x": 1},
                [checkpoint],
            )
            checkpoint.write_text("{}\n", encoding="utf-8")

            compatible = prepare_stage_checkpoint(
                output,
                "test",
                "fp2",
                {"x": 2},
                [checkpoint],
            )
            self.assertFalse(compatible)
            self.assertFalse(checkpoint.exists())
            meta = json.loads((output / ".checkpoint_meta.json").read_text())
            self.assertEqual(meta["fingerprint"], "fp2")


if __name__ == "__main__":
    unittest.main()
