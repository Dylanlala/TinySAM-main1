#!/usr/bin/env python3
"""
Kvasir-SEG dataset loader for YOLO-frozen + TinySAM training.

Loads paired RGB images and binary masks, optionally applying light data
augmentation. Designed to work with TinySAM training loops that expect images
in [0, 1] floats and masks as float tensors.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


@dataclass
class SampleInfo:
    """Metadata returned alongside each tensor sample."""

    image_path: str
    mask_path: str
    image_id: str


class KvasirSegTinySAMDataset(Dataset):
    """
    Kvasir-SEG dataset for YOLO→TinySAM training.

    Args:
        data_dir: Root folder containing `images/` and `masks/`.
        split: One of {"train", "val", "test"}.
        img_size: Target square resolution.
        augment: Whether to apply random spatial/color augmentations.
        split_ratio: Train/val/test ratios used when `index_file` is None.
        split_seed: Seed for deterministic splitting.
        index_file: Optional JSON file specifying sample lists for each split.
                    Expected format: {"train": [...], "val": [...], "test": [...]}
                    where each entry is {"image": "...", "mask": "..."} with paths
                    relative to `data_dir`.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 1024,
        augment: bool = False,
        split_ratio: Sequence[float] = (0.8, 0.1, 0.1),
        split_seed: int = 42,
        index_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'")

        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.split = split
        self.img_size = img_size
        self.augment = augment

        if not self.images_dir.exists() or not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Expected 'images' and 'masks' directories under {self.data_dir}"
            )

        if index_file:
            index_path = Path(index_file)
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_file}")
            with open(index_path, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            if split not in split_data:
                raise KeyError(f"Split '{split}' missing in index file {index_file}")
            items = split_data[split]
            self.samples: List[SampleInfo] = []
            for entry in items:
                img_rel = entry.get("image")
                mask_rel = entry.get("mask")
                if img_rel is None or mask_rel is None:
                    continue
                img_path = self.data_dir / img_rel
                mask_path = self.data_dir / mask_rel
                if not img_path.exists() or not mask_path.exists():
                    continue
                self.samples.append(
                    SampleInfo(
                        image_path=str(img_path),
                        mask_path=str(mask_path),
                        image_id=Path(img_rel).stem,
                    )
                )
        else:
            all_images = sorted(self.images_dir.glob("*.jpg"))
            if not all_images:
                all_images = sorted(self.images_dir.glob("*.png"))
            if not all_images:
                raise RuntimeError(f"No image files found in {self.images_dir}")

            paired: List[SampleInfo] = []
            for img_path in all_images:
                mask_path = self.masks_dir / img_path.name
                if not mask_path.exists():
                    continue
                paired.append(
                    SampleInfo(
                        image_path=str(img_path),
                        mask_path=str(mask_path),
                        image_id=img_path.stem,
                    )
                )

            if not paired:
                raise RuntimeError("Found zero valid (image, mask) pairs.")

            train_ratio, val_ratio, test_ratio = split_ratio
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total

            random.Random(split_seed).shuffle(paired)
            n_total = len(paired)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train_set = paired[:n_train]
            val_set = paired[n_train : n_train + n_val]
            test_set = paired[n_train + n_val :]

            splits: Dict[str, List[SampleInfo]] = {
                "train": train_set,
                "val": val_set if val_set else test_set,
                "test": test_set if test_set else val_set,
            }
            self.samples = splits[split]

        if not self.samples:
            raise RuntimeError(f"No samples available for split '{split}'")

        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")

        # Resize first to maintain alignment
        image = image.resize((self.img_size, self.img_size), Image.BICUBIC)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment and self.split == "train":
            image, mask = self._apply_augmentations(image, mask)

        image_tensor = TF.to_tensor(image)  # [0, 1]
        mask_tensor = torch.from_numpy(
            (np.array(mask, dtype=np.uint8) > 127).astype(np.float32)
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
            "image_id": sample.image_id,
        }

    def _apply_augmentations(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < 0.2:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        if random.random() < 0.5:
            image = self.color_jitter(image)
        return image, mask

