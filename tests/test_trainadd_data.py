import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from data_prep.trainadd_data import build_combined_train_data


class TrainaddDataTests(unittest.TestCase):
    def test_added_image_uses_the_original_mask(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            original, added, masks = root / "original", root / "added", root / "masks"
            for folder in (original, added, masks):
                folder.mkdir()
            name = "0001.png"
            cv2.imwrite(str(original / name), np.full((1024, 1024, 3), 180, np.uint8))
            cv2.imwrite(str(added / name), np.full((1024, 1024, 3), 40, np.uint8))
            cv2.imwrite(str(masks / name), np.full((1024, 1024), 7, np.uint8))
            train_split, add_split = root / "train.txt", root / "trainadd.txt"
            train_split.write_text(name + "\n", encoding="utf-8")
            add_split.write_text(name + "\n", encoding="utf-8")
            dataset = build_combined_train_data(
                (original, masks, train_split), (added, masks, add_split)
            )
            self.assertEqual(len(dataset), 2)
            original_image, original_mask, _ = dataset[0]
            added_image, added_mask, _ = dataset[1]
            self.assertNotEqual(float(original_image.mean()), float(added_image.mean()))
            np.testing.assert_array_equal(original_mask.numpy(), added_mask.numpy())


if __name__ == "__main__":
    unittest.main()
