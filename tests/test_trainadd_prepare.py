import tempfile
import unittest
from pathlib import Path

from data_prep.prepare_trainadd import build_entries


class TrainaddPrepareTests(unittest.TestCase):
    def test_processed_files_keep_original_names_for_shared_masks(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            (root / "agricultural_0001_heavy.png").write_bytes(b"processed")
            entries = build_entries(
                [{"source_name": "0001.png", "processed_file": "agricultural_0001_heavy.png"}],
                {"0001.png"}, root,
            )
            self.assertEqual(entries, [("0001.png", root / "agricultural_0001_heavy.png")])

    def test_rejects_names_outside_original_training_split(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            (root / "val_heavy.png").write_bytes(b"processed")
            with self.assertRaisesRegex(ValueError, "outside original training split"):
                build_entries([{"source_name": "val.png", "processed_file": "val_heavy.png"}],
                              set(), root)


if __name__ == "__main__":
    unittest.main()
