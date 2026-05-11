"""
ImageNet-1k validation subset loader.

Uses a fixed, reproducible set of image IDs for all experiments.
Never trains or tunes on the validation set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


# Fixed development (smoke test) image IDs — first 100 alphabetically from val set
# These are replaced by actual IDs after first run on Colab
DEV_SUBSET_SIZE = 100
PILOT_SUBSET_SIZE = 100  # different fixed IDs from dev
EVAL_SUBSET_SIZE = 1000  # minimum for final evaluation


class ImageNetValSubset(Dataset):
    """
    A fixed, reproducible subset of the ImageNet-1k validation set.

    Args:
        root: path to ImageNet val directory (contains synset subdirectories)
        image_ids_file: JSON file with list of (synset, filename) pairs.
                        If None, uses the first N images found (for bootstrapping).
        n: number of images to use (only when image_ids_file is None)
        transform: torchvision transform to apply
    """

    def __init__(
        self,
        root: str | Path,
        image_ids_file: Optional[str | Path] = None,
        n: int = DEV_SUBSET_SIZE,
        transform=None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"ImageNet val directory not found: {self.root}\n"
                "Set IMAGENET_VAL_PATH environment variable or pass root explicitly."
            )

        if image_ids_file is not None:
            with open(image_ids_file) as f:
                self.samples = json.load(f)
        else:
            self.samples = self._discover_samples(n)

        self.transform = transform

    def _discover_samples(self, n: int) -> list[dict]:
        """Discover first n images from val directory, sorted for reproducibility."""
        samples = []
        synset_dirs = sorted(self.root.iterdir())
        for synset_dir in synset_dirs:
            if not synset_dir.is_dir():
                continue
            for img_path in sorted(synset_dir.iterdir()):
                if img_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    samples.append({
                        "synset": synset_dir.name,
                        "filename": img_path.name,
                        "path": str(img_path),
                    })
                    if len(samples) >= n:
                        return samples
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        path = sample.get("path") or str(self.root / sample["synset"] / sample["filename"])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "synset": sample["synset"],
            "filename": sample.get("filename", ""),
            "image_id": idx,
        }

    def save_image_ids(self, path: str | Path) -> None:
        """Save the image ID list for reproducibility."""
        with open(path, "w") as f:
            json.dump(self.samples, f, indent=2)


def load_imagenet_subset(
    root: Optional[str] = None,
    image_ids_file: Optional[str] = None,
    n: int = DEV_SUBSET_SIZE,
    transform=None,
) -> ImageNetValSubset:
    """
    Load an ImageNet subset, checking IMAGENET_VAL_PATH env var if root not given.
    """
    if root is None:
        root = os.environ.get("IMAGENET_VAL_PATH")
    if root is None:
        raise EnvironmentError(
            "ImageNet val path not provided. Set IMAGENET_VAL_PATH or pass root= explicitly."
        )
    return ImageNetValSubset(root=root, image_ids_file=image_ids_file, n=n, transform=transform)
