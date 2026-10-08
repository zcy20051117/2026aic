from __future__ import annotations

import numpy as np

from aic_dino_m2f.metrics.confusion import ConfusionMatrix8


def test_global_confusion_counts_fp_for_class_absent_from_one_image() -> None:
    meter = ConfusionMatrix8()
    meter.update(
        np.array([[0, 0], [1, 1]], dtype=np.uint8),
        np.array([[0, 2], [1, 1]], dtype=np.uint8),
    )
    meter.update(
        np.array([[2, 2], [255, 0]], dtype=np.uint8),
        np.array([[2, 0], [7, 0]], dtype=np.uint8),
    )
    result = meter.compute(require_all_classes=False)
    assert meter.matrix.dtype == np.int64
    assert meter.matrix.sum() == 7
    assert meter.matrix[0, 2] == 1
    assert meter.matrix[2, 0] == 1
    assert np.isclose(result["per_class_iou"][0], 2 / 4)
    assert np.isclose(result["per_class_iou"][1], 1.0)
    assert np.isclose(result["per_class_iou"][2], 1 / 3)


def test_confusion_is_invariant_to_batch_partitioning() -> None:
    gt = np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.uint8)
    pred = np.array([[0, 0], [2, 3], [4, 0], [6, 7]], dtype=np.uint8)
    whole = ConfusionMatrix8()
    whole.update(gt, pred)
    split = ConfusionMatrix8()
    split.update(gt[:2], pred[:2])
    split.update(gt[2:], pred[2:])
    np.testing.assert_array_equal(whole.matrix, split.matrix)


def test_confusion_reports_per_class_precision_and_recall() -> None:
    meter = ConfusionMatrix8()
    meter.update(
        np.array([[4, 4, 0], [0, 1, 1]], dtype=np.uint8),
        np.array([[4, 0, 4], [0, 1, 4]], dtype=np.uint8),
    )

    result = meter.compute(require_all_classes=False)

    # Barren (internal class 4): TP=1, predicted=3, ground-truth=2.
    assert np.isclose(result["per_class_precision"][4], 1 / 3)
    assert np.isclose(result["per_class_recall"][4], 1 / 2)
    assert np.isclose(result["per_class_precision_by_name"]["Barren"], 1 / 3)
    assert np.isclose(result["per_class_recall_by_name"]["Barren"], 1 / 2)
