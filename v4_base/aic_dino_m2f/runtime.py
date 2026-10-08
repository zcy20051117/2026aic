from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from aic_dino_m2f.config import config_sha256, load_and_validate_config
from aic_dino_m2f.data.dataset import (
    AICSegmentationDataset,
    build_clean_hard_training_dataset,
    validate_split_lists,
)
from aic_dino_m2f.data.augmentation import build_train_barren_augmenter
from aic_dino_m2f.data.sampling import build_barren_weighted_sampler, compute_barren_ratios
from aic_dino_m2f.engine.checkpoint import load_training_checkpoint
from aic_dino_m2f.engine.manifest import sha256_file
from aic_dino_m2f.engine.optim import (
    PolynomialWarmupScheduler,
    build_adamw,
    formal_optimizer_steps_per_epoch,
)
from aic_dino_m2f.engine.train_epoch import collate_samples
from aic_dino_m2f.losses.criterion import SetCriterion
from aic_dino_m2f.losses.matcher import HungarianMatcher
from aic_dino_m2f.models.adapter import TrainableDINOv3Adapter
from aic_dino_m2f.models.backbone import load_frozen_dinov3_vitl16
from aic_dino_m2f.models.mask2former import TrainableMask2FormerHead
from aic_dino_m2f.models.model import DINOv3Mask2Former


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(config: Mapping[str, object]) -> DINOv3Mask2Former:
    model_config = config["model"]
    backbone = load_frozen_dinov3_vitl16(Path(model_config["backbone_weights"]))
    adapter = TrainableDINOv3Adapter(
        backbone,
        interaction_indexes=list(model_config["interaction_indexes"]),
        pretrain_size=int(model_config["adapter_pretrain_size"]),
        conv_inplane=int(model_config["adapter_conv_inplane"]),
        deform_num_heads=int(model_config["adapter_deform_heads"]),
        drop_path_rate=float(model_config["adapter_drop_path"]),
        with_cp=False,
        convert_sync_batchnorm=True,
    )
    decoder = TrainableMask2FormerHead(
        input_channels=int(backbone.embed_dim),
        num_classes=int(model_config["num_classes"]),
        hidden_dim=int(model_config["hidden_dim"]),
        mask_dim=int(model_config["mask_dim"]),
        conv_dim=int(model_config["conv_dim"]),
        pixel_decoder_encoder_layers=int(model_config["pixel_decoder_encoder_layers"]),
        pixel_decoder_attention_heads=int(model_config["pixel_decoder_attention_heads"]),
        pixel_decoder_ffn_dim=int(model_config["pixel_decoder_ffn_dim"]),
        transformer_decoder_layers=int(model_config["transformer_decoder_layers"]),
        transformer_decoder_attention_heads=int(model_config["transformer_decoder_attention_heads"]),
        transformer_decoder_ffn_dim=int(model_config["transformer_decoder_ffn_dim"]),
        num_queries=int(model_config["num_queries"]),
        dropout=float(model_config["dropout"]),
        common_stride=int(model_config["common_stride"]),
    )
    return DINOv3Mask2Former(adapter, decoder)


def build_criterion(config: Mapping[str, object]) -> SetCriterion:
    loss = config["loss"]
    matcher = HungarianMatcher(
        class_cost=float(loss["class_weight"]), mask_cost=float(loss["mask_weight"]),
        dice_cost=float(loss["dice_weight"]), num_points=int(loss["num_points"]),
        oversample_ratio=float(loss["oversample_ratio"]),
        importance_sample_ratio=float(loss["importance_sample_ratio"]),
    )
    return SetCriterion(
        num_classes=8, matcher=matcher, class_weight=float(loss["class_weight"]),
        mask_weight=float(loss["mask_weight"]), dice_weight=float(loss["dice_weight"]),
        no_object_weight=float(loss["no_object_weight"]), num_points=int(loss["num_points"]),
        oversample_ratio=float(loss["oversample_ratio"]),
        importance_sample_ratio=float(loss["importance_sample_ratio"]),
        semantic_class_weights=loss.get("semantic_class_weights"),
    )


def checkpoint_metadata(config: Mapping[str, object]) -> dict[str, object]:
    data = config["data"]
    model = config["model"]
    return {
        "config_sha256": config_sha256(config),
        "backbone_identity": model["backbone"],
        "backbone_sha256": sha256_file(Path(model["backbone_weights"])),
        "train_split_sha256": sha256_file(Path(data["train_list"])),
        "hard_split_sha256": sha256_file(Path(data["hard_list"])),
        "val_split_sha256": sha256_file(Path(data["val_list"])),
        "label_mapping": {"0": 255, **{str(index): index - 1 for index in range(1, 9)}},
        "input_size": int(data["image_size"]),
        "effective_batch": int(config["training"]["target_effective_batch"]),
        "architecture": "frozen-dinov3-vitl16-adapter-mask2former-256d-v1",
    }


def _datasets(config: Mapping[str, object]):
    data = config["data"]
    image_size = (int(data["image_size"]),) * 2
    validate_split_lists(
        Path(data["train_list"]), Path(data["val_list"]),
        Path(data["images_dir"]), Path(data["masks_dir"]),
        expected_total=int(data["original_count"]) + int(data["val_count"]),
    )
    augmentation = config.get("augmentation", {})
    if not isinstance(augmentation, Mapping):
        raise ValueError("augmentation configuration must be a mapping")
    train_transform = build_train_barren_augmenter(augmentation) if augmentation.get("enabled", False) else None
    train_dataset = build_clean_hard_training_dataset(
        Path(data["train_list"]), Path(data["val_list"]), Path(data["hard_list"]),
        Path(data["images_dir"]), Path(data["hard_images_dir"]), Path(data["masks_dir"]),
        expected_size=image_size, expected_original=int(data["original_count"]),
        expected_hard=int(data["hard_count"]), pair_transform=train_transform,
    )
    val_dataset = AICSegmentationDataset(
        Path(data["val_list"]), Path(data["images_dir"]), Path(data["masks_dir"]),
        expected_size=image_size,
    )
    if len(train_dataset) != int(data["train_count"]) or len(val_dataset) != int(data["val_count"]):
        raise ValueError("Configured training/validation counts do not match cleaned splits")
    return train_dataset, val_dataset


def build_training_runtime(
    config: Mapping[str, object], *, train_limit: int | None = None, physical_batch_size: int | None = None
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; enable a GPU before model smoke/training")
    seed = int(config["seed"])
    seed_everything(seed)
    train_dataset, val_dataset = _datasets(config)
    if train_limit is not None:
        train_dataset = Subset(train_dataset, range(min(train_limit, len(train_dataset))))
    batch_size = int(physical_batch_size or config["training"]["physical_batch_size"])
    target_batch = int(config["training"]["target_effective_batch"])
    if target_batch % batch_size:
        raise ValueError("Physical batch must divide target effective batch")
    accumulation_steps = target_batch // batch_size
    generator = torch.Generator().manual_seed(seed)
    workers = int(config["training"]["num_workers"])
    sampling = config.get("sampling", {})
    if not isinstance(sampling, Mapping):
        raise ValueError("sampling configuration must be a mapping")
    sampler = None
    if sampling.get("barren_weighted", False):
        if isinstance(train_dataset, Subset):
            ratios = compute_barren_ratios(
                train_dataset.dataset,
                indices=train_dataset.indices,
                show_progress=True,
            )
        else:
            ratios = compute_barren_ratios(train_dataset, show_progress=True)
        sampler = build_barren_weighted_sampler(ratios, seed=seed, num_samples=len(train_dataset))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, drop_last=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, worker_init_fn=_seed_worker,
        generator=generator, collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(config["validation"]["batch_size"]), shuffle=False,
        drop_last=False, num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        worker_init_fn=_seed_worker, collate_fn=collate_samples,
    )
    device = torch.device("cuda", 0)
    model = build_model(config).to(device)
    criterion = build_criterion(config).to(device)
    optimizer = build_adamw(
        model, learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer_steps_per_epoch = formal_optimizer_steps_per_epoch(
        sample_count=int(config["data"]["train_count"]),
        physical_batch_size=batch_size,
        accumulation_steps=accumulation_steps,
    )
    scheduler = PolynomialWarmupScheduler(
        optimizer,
        total_steps=optimizer_steps_per_epoch * int(config["training"]["max_epochs"]),
        warmup_steps=optimizer_steps_per_epoch * int(config["training"]["warmup_epochs"]),
        warmup_start_factor=float(config["training"]["warmup_start_factor"]),
        power=float(config["training"]["poly_power"]),
        min_learning_rate=float(config["training"]["min_learning_rate"]),
    )
    return {
        "model": model, "criterion": criterion, "optimizer": optimizer, "scheduler": scheduler,
        "train_loader": train_loader, "val_loader": val_loader, "device": device,
        "accumulation_steps": accumulation_steps, "checkpoint_metadata": checkpoint_metadata(config),
    }


def load_inference_runtime(config_path: Path, checkpoint_path: Path):
    config = load_and_validate_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 1024x1024 inference")
    device = torch.device("cuda", 0)
    model = build_model(config).to(device)
    load_training_checkpoint(
        checkpoint_path, model=model, optimizer=None, scheduler=None, scaler=None,
        expected_metadata=checkpoint_metadata(config), restore_rng=False,
    )
    return model, device, config


def load_evaluation_runtime(config_path: Path, checkpoint_path: Path):
    model, device, config = load_inference_runtime(config_path, checkpoint_path)
    _, val_dataset = _datasets(config)
    loader = DataLoader(
        val_dataset, batch_size=int(config["validation"]["batch_size"]), shuffle=False,
        num_workers=int(config["training"]["num_workers"]), pin_memory=True,
        collate_fn=collate_samples,
    )
    return model, loader, device, config
