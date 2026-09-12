"""GenFace dataset loader interface.

# ASSUMPTION_FROM_PAPER_GAP:
# The paper uses GenFace benchmark (Zhang et al., 2024), but the exact directory structure
# and metadata manifest layout are not specified in the paper.
# This module implements a flexible manifest/folder-based dataset loader.
"""

import os
from typing import Any, Callable, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image

from datasets.prompt_generator import FineGrainedTextGenerator
from datasets.mask_generator import generate_ground_truth_mask
from datasets.transforms import get_transforms


class GenFaceDataset(Dataset):
    """Configurable dataset loader for the GenFace face forgery benchmark."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        max_text_tokens: int = 308,
        vocab_size: int = 49408,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.max_text_tokens = max_text_tokens
        self.vocab_size = vocab_size
        self.transform = transform if transform is not None else get_transforms(image_size)

        self.prompt_generator = FineGrainedTextGenerator()
        self.samples: List[Dict[str, Any]] = []

        if not os.path.exists(root_dir):
            # Not an error at init if using dummy dataset, but clearly documented
            pass
        else:
            self._load_dataset()

    def _load_dataset(self) -> None:
        """Scan root_dir or load manifest if present."""
        # Configurable scan for images
        manifest_file = os.path.join(self.root_dir, f"{self.split}.json")
        if os.path.exists(manifest_file):
            import json
            with open(manifest_file, "r") as f:
                self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.samples:
            raise IndexError("GenFaceDataset has no samples loaded. Ensure root_dir exists.")

        item = self.samples[idx]
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        is_fake = item.get("is_fake", True)
        label_one_hot = torch.tensor([0.0, 1.0] if is_fake else [1.0, 0.0], dtype=torch.float32)
        label_idx = torch.tensor(1 if is_fake else 0, dtype=torch.long)

        # Source image if available
        has_source = False
        if "source_path" in item and item["source_path"] and os.path.exists(item["source_path"]):
            source_img = Image.open(item["source_path"]).convert("RGB")
            source_image_tensor = self.transform(source_img)
            has_source = True
        else:
            source_image_tensor = torch.zeros_like(image_tensor)

        # Ground truth mask
        forgery_type = item.get("forgery_type", "EFS" if is_fake else "Real")
        mask = generate_ground_truth_mask(
            is_fake=is_fake,
            forgery_type=forgery_type,
            fake_image=image_tensor,
            source_image=source_image_tensor if has_source else None,
            image_size=self.image_size,
        )

        prompts = self.prompt_generator.generate_prompts(
            is_fake=is_fake,
            forgery_type=forgery_type if is_fake else None,
            generator=item.get("generator"),
            family=item.get("family"),
        )

        return {
            "image": image_tensor,
            "label": label_one_hot,
            "label_idx": label_idx,
            "mask": mask,
            "prompt": prompts["full_prompt"],
            "prompts_dict": prompts,
            "generator": item.get("generator", "Unknown"),
            "forgery_type": forgery_type,
            "source_image": source_image_tensor,
            "has_source": has_source,
        }
