#!/usr/bin/env python3
"""
dataset.py - PyTorch Dataset & DataLoader for MFVLR / GenFace Reproduction

Implements MFVLRDataset:
- Output sample format:
  {
      "image": torch.Tensor,       # [3, 224, 224], float32
      "mask": torch.Tensor,        # [1, 224, 224], float32, binary {0.0, 1.0}
      "label": torch.Tensor,       # int64 (0: Real, 1: Fake)
      "forgery_type": str,         # 'REAL', 'EFS', 'AM', 'FS'
      "architecture": str,         # 'REAL', 'Diffusion', 'GAN'
      "generator": str,            # e.g. 'DDPM', 'DiffFace', 'Real'
      "sample_id": str,
      "prompts": {
          "L1": str,
          "L2": str,
          "L3": str,
          "L4": str,
      }
  }
- Synchronized augmentations (flips/crops applied identically to both image and mask).
- Resizing guaranteed at 224x224.
- DataLoader factory function `create_dataloader`.
"""

import csv
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from prompt_generator import PromptGenerator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MFVLRDataset")

# Default ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SynchronizedTransform:
    """Applies geometric augmentations synchronously to both image and mask."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        horizontal_flip_prob: float = 0.5,
        normalize: bool = True,
        mean: List[float] = IMAGENET_MEAN,
        std: List[float] = IMAGENET_STD,
    ) -> None:
        self.target_size = target_size
        self.horizontal_flip_prob = horizontal_flip_prob
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Deterministic resize
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        # 2. Synchronized random horizontal flip
        if self.horizontal_flip_prob > 0 and random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 3. Convert to tensor
        # Image: [3, H, W], float32 in [0.0, 1.0]
        img_tensor = TF.to_tensor(image)
        if self.normalize:
            img_tensor = TF.normalize(img_tensor, mean=self.mean, std=self.std)

        # Mask: [1, H, W], float32 in {0.0, 1.0}
        mask_np = np.array(mask, dtype=np.float32)
        if mask_np.ndim == 2:
            mask_np = np.expand_dims(mask_np, axis=0)
        elif mask_np.ndim == 3:
            mask_np = mask_np[:1, :, :]  # Take single channel

        # Binarize: > 127 mapped to 1.0, else 0.0
        mask_tensor = torch.from_numpy((mask_np > 127.5).astype(np.float32))

        return img_tensor, mask_tensor


class MFVLRDataset(Dataset):
    """PyTorch Dataset for MFVLR / GenFace reproduction."""

    def __init__(
        self,
        metadata_path: Union[str, Path],
        dataset_root: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        is_training: bool = False,
    ) -> None:
        """
        Args:
            metadata_path: Path to metadata CSV (e.g. all.csv, train.csv, val.csv, test.csv).
            dataset_root: Root path of MFVLR dataset (inferred if None).
            split: Filter rows by split ('train', 'val', 'test') if metadata has multiple splits.
            transform: Custom synchronized transform callable (img, mask) -> (img_t, mask_t).
            target_size: (width, height), default (224, 224).
            normalize: Whether to apply ImageNet normalization to image tensor.
            is_training: If True and transform is None, enables random horizontal flips.
        """
        self.metadata_path = Path(metadata_path).resolve()
        self.dataset_root = (
            Path(dataset_root).resolve()
            if dataset_root
            else self.metadata_path.parent.parent
        )
        self.target_size = target_size

        if transform is not None:
            self.transform = transform
        else:
            flip_p = 0.5 if is_training else 0.0
            self.transform = SynchronizedTransform(
                target_size=target_size,
                horizontal_flip_prob=flip_p,
                normalize=normalize,
            )

        self.samples: List[Dict[str, Any]] = []
        self._load_samples(split)
        self.prompt_gen = PromptGenerator()

    def _load_samples(self, split: Optional[str]) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split and row.get("split", "").strip().lower() != split.lower():
                    continue
                self.samples.append(row)

        logger.info(
            "Loaded %d samples from %s (split filter: %s)",
            len(self.samples),
            self.metadata_path.name,
            split or "None",
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.samples[idx]
        sample_id = row.get("sample_id", f"sample_{idx}")

        # Resolve paths
        img_rel = row.get("image_path", "")
        mask_rel = row.get("mask_path", "")

        img_path = self.dataset_root / img_rel
        mask_path = self.dataset_root / mask_rel

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path} (sample_id={sample_id})")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path} (sample_id={sample_id})")

        # Load image & mask as PIL
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        with Image.open(mask_path) as mask:
            mask = mask.convert("L")

        # Apply synchronized transform
        img_tensor, mask_tensor = self.transform(img, mask)

        # Label & metadata
        label = int(row.get("label", 0))
        forgery_type = row.get("forgery_type", "REAL")
        architecture = row.get("architecture", "REAL")
        generator = row.get("generator", "Real")

        # Prompts L1-L4: use CSV values if present, else fallback to PromptGenerator
        l1 = row.get("L1") or row.get("prompt_L1")
        l2 = row.get("L2") or row.get("prompt_L2")
        l3 = row.get("L3") or row.get("prompt_L3")
        l4 = row.get("L4") or row.get("prompt_L4")

        if not all([l1, l2, l3, l4]):
            p = self.prompt_gen.generate(
                label=label,
                forgery_type=forgery_type,
                architecture=architecture,
                generator=generator,
            )
            l1, l2, l3, l4 = p["L1"], p["L2"], p["L3"], p["L4"]

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "forgery_type": forgery_type,
            "architecture": architecture,
            "generator": generator,
            "sample_id": sample_id,
            "prompts": {
                "L1": l1,
                "L2": l2,
                "L3": l3,
                "L4": l4,
            },
        }


def create_dataloader(
    metadata_path: Union[str, Path],
    dataset_root: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    is_training: bool = False,
    normalize: bool = True,
) -> DataLoader:
    """Creates a PyTorch DataLoader for MFVLR dataset."""
    dataset = MFVLRDataset(
        metadata_path=metadata_path,
        dataset_root=dataset_root,
        split=split,
        normalize=normalize,
        is_training=is_training,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    print("MFVLRDataset module. Import into training scripts or use create_dataloader().")
