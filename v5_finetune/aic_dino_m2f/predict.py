from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from aic_dino_m2f.data.transforms import normalize_rgb_image
from aic_dino_m2f.models.postprocess import semantic_scores


class ImageOnlyDataset(Dataset):
    def __init__(self, image_dir: Path, *, expected_size: tuple[int, int] = (1024, 1024)) -> None:
        self.root = Path(image_dir).resolve(strict=True)
        self.paths = sorted(path for path in self.root.iterdir() if path.suffix.lower() == ".png")
        self.expected_size = expected_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            if (image.height, image.width) != self.expected_size:
                raise ValueError(f"Image {path.name} has unexpected size")
            tensor = normalize_rgb_image(image)
        return tensor, path.name


@torch.inference_mode()
def predict_images(*, model, loader, output_dir: Path, device: torch.device, query_chunk: int) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    progress = tqdm(loader, total=len(loader), desc="predict images", dynamic_ncols=True)
    written = 0
    for images, names in progress:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        internal = semantic_scores(
            outputs["pred_logits"], outputs["pred_masks"], output_size=(1024, 1024), query_chunk=query_chunk
        ).argmax(dim=1)
        official = (internal + 1).to(torch.uint8).cpu().numpy()
        for name, mask in zip(names, official, strict=True):
            Image.fromarray(np.asarray(mask, dtype=np.uint8)).save(output / name)
            written += 1
        progress.set_postfix(files=written, refresh=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-image AIC inference; no ground truth is accepted")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from aic_dino_m2f.runtime import load_inference_runtime
    from aic_dino_m2f.engine.manifest import assert_project_path

    model, device, config = load_inference_runtime(args.config, args.checkpoint)
    dataset = ImageOnlyDataset(args.image_dir)
    loader = DataLoader(dataset, batch_size=int(config["validation"]["batch_size"]), shuffle=False)
    output = assert_project_path(args.output_dir, Path(config["project_root"]))
    predict_images(model=model, loader=loader, output_dir=output, device=device, query_chunk=int(config["validation"]["query_chunk"]))


if __name__ == "__main__":
    main()
