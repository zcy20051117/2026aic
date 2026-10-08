# Checkpoint lineage

V4 starts from the DINOv3 pretrained weights named by `DINOV3_WEIGHTS` and selects V4 `best.pth` by clean-validation mIoU. V5 initializes its trainable weights from that V4 checkpoint. The `resume15` stage restores V5 `last.pth`, including optimizer, scheduler, and RNG state, then chooses the final `best.pth` by the same clean-validation mIoU criterion.
