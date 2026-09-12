"""Generic Manifest-driven Dataset loader for MFVLR.

PAPER_SPECIFIED:
- Image shape: [3, 224, 224] in [0, 1]
- Localization mask shape: [224, 224] in {0, 1}
- Hierarchical text prompts: L1 to L4 (token sequence n = 308, s = 49408)
- Real mask = 0 (all zeros), EFS mask = 1 (all ones), AM/FS mask derived from |I_fake - I_source| > 0.1.

ASSUMPTION_FROM_PAPER_GAP:
- Manifest formats supported: JSONL (.jsonl), JSON (.json), CSV (.csv).
- Configurable label mapping: real_class_index (default: 0), fake_class_index (default: 1).
- Paths resolved relative to dataset_root.
"""

import csv
import json
import os
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets.prompt_generator import FineGrainedTextGenerator
from datasets.mask_generator import generate_ground_truth_mask
from datasets.tokenizer import MFVLRTokenizer, tokenize
from datasets.transforms import get_transforms


class MFVLRDataset(Dataset):
    """Manifest-driven PyTorch dataset for MFVLR face forgery detection and localization."""

    def __init__(
        self,
        manifest_path: str,
        dataset_root: Optional[str] = None,
        image_size: int = 224,
        max_text_tokens: int = 308,
        vocab_size: int = 49408,
        real_class_index: int = 0,
        fake_class_index: int = 1,
        transform: Optional[Callable] = None,
        require_prompts: bool = True,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.dataset_root = dataset_root or os.path.dirname(os.path.abspath(manifest_path))
        self.image_size = image_size
        self.max_text_tokens = max_text_tokens
        self.vocab_size = vocab_size
        self.real_class_index = real_class_index
        self.fake_class_index = fake_class_index
        self.transform = transform if transform is not None else get_transforms(image_size)
        self.require_prompts = require_prompts

        self.prompt_generator = FineGrainedTextGenerator()
        self.tokenizer = MFVLRTokenizer(max_tokens=max_text_tokens, vocab_size=vocab_size)
        self.samples: List[Dict[str, Any]] = self._load_manifest(manifest_path)

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        if path is None or str(path).strip() == "":
            return None
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.dataset_root, path))

    def _load_manifest(self, path: str) -> List[Dict[str, Any]]:
        """Load manifest from JSONL, JSON, or CSV file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Manifest file not found at: {path}")

        samples: List[Dict[str, Any]] = []

        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record = json.loads(line_str)
                        samples.append(record)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Malformed JSON on line {line_idx + 1} of {path}: {e}")

        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict) and "samples" in data:
                    samples = data["samples"]
                else:
                    raise ValueError(f"JSON manifest must contain a list of records or a 'samples' key: {path}")

        elif path.endswith(".csv"):
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(dict(row))
        else:
            raise ValueError(f"Unsupported manifest file extension for: {path}. Supported: .jsonl, .json, .csv")

        if len(samples) == 0:
            raise ValueError(f"Manifest at {path} contains 0 valid sample records.")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # 1. Resolve image path
        raw_img_path = item.get("image_path") or item.get("img_path") or item.get("image")
        if not raw_img_path:
            raise KeyError(f"Sample at index {idx} is missing 'image_path' field.")

        full_img_path = self._resolve_path(raw_img_path)
        if not os.path.exists(full_img_path):
            raise FileNotFoundError(f"Image file not found at: {full_img_path}")

        image = Image.open(full_img_path).convert("RGB")
        image_tensor = self.transform(image)  # [3, 224, 224] in [0, 1]

        # 2. Determine binary label and forgery type
        # Check label representation
        is_fake: bool
        if "is_fake" in item:
            val = item["is_fake"]
            is_fake = bool(val) if isinstance(val, bool) else (str(val).lower() in ("true", "1", "fake"))
        elif "label" in item:
            val = item["label"]
            if isinstance(val, (int, float)):
                is_fake = (int(val) == self.fake_class_index)
            else:
                is_fake = (str(val).lower() in ("fake", "1", "true"))
        elif "class_index" in item:
            is_fake = (int(item["class_index"]) == self.fake_class_index)
        else:
            # Default fallback to real if manipulation_type is real, else fake
            forgery_type_str = str(item.get("manipulation_type", item.get("forgery_type", ""))).upper()
            is_fake = (forgery_type_str not in ("REAL", "0", ""))

        target_class_idx = self.fake_class_index if is_fake else self.real_class_index
        label_tensor = torch.tensor(target_class_idx, dtype=torch.long)

        # 3. Forgery type and source image for mask generation
        forgery_type = item.get("manipulation_type") or item.get("forgery_type")
        if forgery_type is not None:
            forgery_type = str(forgery_type).upper()
            if forgery_type in ("REAL", "0"):
                forgery_type = "Real"
            elif forgery_type in ("EFS", "ENTIRE_SYNTHESIS", "ENTIRE_FACE_SYNTHESIS"):
                forgery_type = "EFS"
            elif forgery_type in ("AM", "ATTRIBUTE_MANIPULATION"):
                forgery_type = "AM"
            elif forgery_type in ("FS", "FACE_SWAPPING", "FACE_SWAP"):
                forgery_type = "FS"
        else:
            forgery_type = "EFS" if is_fake else "Real"

        # Resolve source image if required for AM/FS
        source_image_tensor: Optional[torch.Tensor] = None
        raw_source_path = item.get("source_image_path") or item.get("source_path") or item.get("source")

        if raw_source_path:
            full_source_path = self._resolve_path(raw_source_path)
            if not os.path.exists(full_source_path):
                raise FileNotFoundError(f"Source image specified but not found at: {full_source_path}")
            source_img = Image.open(full_source_path).convert("RGB")
            source_image_tensor = self.transform(source_img)

        # Generate ground truth localization mask
        mask_tensor = generate_ground_truth_mask(
            is_fake=is_fake,
            forgery_type=forgery_type,
            fake_image=image_tensor,
            source_image=source_image_tensor,
            image_size=self.image_size,
        ).long()  # [224, 224] long class indices {0, 1}

        # 4. Token IDs for prompt
        token_ids: torch.Tensor
        if "token_ids" in item and isinstance(item["token_ids"], list):
            token_ids = torch.tensor(item["token_ids"], dtype=torch.long)
            if token_ids.shape != (self.max_text_tokens,):
                raise ValueError(f"Sample token_ids must have length {self.max_text_tokens}, got {len(token_ids)}")
        else:
            prompt_str: str
            if "prompt" in item and isinstance(item["prompt"], str):
                prompt_str = item["prompt"]
            else:
                prompts_dict = self.prompt_generator.generate_prompts(
                    is_fake=is_fake,
                    forgery_type=forgery_type if is_fake else None,
                    generator=item.get("generator"),
                    family=item.get("family"),
                )
                prompt_str = prompts_dict["full_prompt"]

            token_ids = self.tokenizer.encode(prompt_str)

        return {
            "image": image_tensor,
            "token_ids": token_ids,
            "label": label_tensor,
            "class_target": label_tensor,
            "mask": mask_tensor,
            "mask_target": mask_tensor,
            "is_fake": is_fake,
            "forgery_type": forgery_type,
            "image_path": full_img_path,
        }
