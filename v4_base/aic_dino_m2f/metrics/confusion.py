from __future__ import annotations

import numpy as np

from aic_dino_m2f.data.constants import CLASS_NAMES, IGNORE_INTERNAL_ID, NUM_CLASSES


class ConfusionMatrix8:
    def __init__(self) -> None:
        self.matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    def update(self, ground_truth: np.ndarray, prediction: np.ndarray) -> None:
        gt = np.asarray(ground_truth)
        pred = np.asarray(prediction)
        if gt.shape != pred.shape:
            raise ValueError(f"GT/prediction shape mismatch: {gt.shape} versus {pred.shape}")
        valid = gt != IGNORE_INTERNAL_ID
        valid_gt = gt[valid].astype(np.int64, copy=False)
        valid_pred = pred[valid].astype(np.int64, copy=False)
        if np.any((valid_gt < 0) | (valid_gt >= NUM_CLASSES)):
            raise ValueError("Ground truth contains an invalid scored class")
        if np.any((valid_pred < 0) | (valid_pred >= NUM_CLASSES)):
            raise ValueError("Prediction contains an invalid scored class")
        counts = np.bincount(
            valid_gt * NUM_CLASSES + valid_pred,
            minlength=NUM_CLASSES * NUM_CLASSES,
        ).reshape(NUM_CLASSES, NUM_CLASSES)
        self.matrix += counts.astype(np.int64, copy=False)

    def compute(self, *, require_all_classes: bool = True) -> dict[str, object]:
        matrix = self.matrix.astype(np.float64)
        true_positive = np.diag(matrix)
        predicted = matrix.sum(axis=0)
        ground_truth = matrix.sum(axis=1)
        union = matrix.sum(axis=1) + matrix.sum(axis=0) - true_positive
        if require_all_classes and np.any(union == 0):
            missing = [CLASS_NAMES[index] for index in np.flatnonzero(union == 0)]
            raise RuntimeError(f"Global validation has zero union for classes: {missing}")
        iou = np.divide(
            true_positive,
            union,
            out=np.full(NUM_CLASSES, np.nan, dtype=np.float64),
            where=union > 0,
        )
        precision = np.divide(
            true_positive,
            predicted,
            out=np.full(NUM_CLASSES, np.nan, dtype=np.float64),
            where=predicted > 0,
        )
        recall = np.divide(
            true_positive,
            ground_truth,
            out=np.full(NUM_CLASSES, np.nan, dtype=np.float64),
            where=ground_truth > 0,
        )
        total = matrix.sum()
        pixel_accuracy = float(true_positive.sum() / total) if total else float("nan")
        return {
            "per_class_iou": iou,
            "per_class_iou_by_name": {
                name: float(iou[index]) for index, name in enumerate(CLASS_NAMES)
            },
            "per_class_precision": precision,
            "per_class_precision_by_name": {
                name: float(precision[index]) for index, name in enumerate(CLASS_NAMES)
            },
            "per_class_recall": recall,
            "per_class_recall_by_name": {
                name: float(recall[index]) for index, name in enumerate(CLASS_NAMES)
            },
            "miou": float(iou.mean()) if require_all_classes else float(np.nanmean(iou)),
            "pixel_accuracy": pixel_accuracy,
            "scored_pixels": int(total),
        }
