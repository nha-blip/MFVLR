"""Dummy dataset for MFVLR pipeline and unit tests without requiring real GenFace data.

PAPER_SPECIFIED constraints respected:
- image shape: [3, 224, 224]
- mask shape: [224, 224]
- label: two-class one-hot [0, 1] or [1, 0], or integer label
- token count: n = 308, vocabulary s = 49408
"""

import random
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset

from datasets.prompt_generator import FineGrainedTextGenerator
from datasets.mask_generator import generate_ground_truth_mask


class DummyMFVLRDataset(Dataset):
    """Synthetic dataset generating synthetic image-text pairs matching GenFace schema."""

    def __init__(
        self,
        num_samples: int = 32,
        image_size: int = 224,
        max_text_tokens: int = 308,
        vocab_size: int = 49408,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_text_tokens = max_text_tokens
        self.vocab_size = vocab_size

        self.prompt_generator = FineGrainedTextGenerator()
        self.generators = [
            ("DDPM", "EFS", "diffusion"),
            ("LatDiff", "EFS", "diffusion"),
            ("CollDiff", "EFS", "diffusion"),
            ("DiffFace", "FS", "diffusion"),
            ("Diffae", "AM", "diffusion"),
            ("StyleGAN3", "EFS", "gan"),
            ("FSLSD", "FS", "gan"),
            ("FaceSwapper", "FS", "gan"),
            ("LatentTransformer", "AM", "gan"),
            ("IA-FaceS", "AM", "gan"),
        ]

        # Deterministic sample metadata list
        rng = random.Random(seed)
        self.samples_meta: List[Dict[str, Any]] = []
        for i in range(num_samples):
            is_fake = (i % 2 == 1)  # Balanced real and fake
            if is_fake:
                gen_info = rng.choice(self.generators)
                gen_name, forgery_type, family = gen_info
            else:
                gen_name, forgery_type, family = "Real", "Real", "Real"

            self.samples_meta.append({
                "is_fake": is_fake,
                "generator": gen_name,
                "forgery_type": forgery_type,
                "family": family,
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.samples_meta[idx]
        is_fake = meta["is_fake"]

        # Synthetic image [3, 224, 224] in [0, 1]
        # Use deterministic generator seeded per item
        g = torch.Generator().manual_seed(idx + 1000)
        image = torch.rand((3, self.image_size, self.image_size), generator=g, dtype=torch.float32)

        # Label: one-hot [2]
        # PAPER_SPECIFIED: y in {[1, 0]^T, [0, 1]^T} where 0=Real, 1=Fake
        if is_fake:
            label_one_hot = torch.tensor([0.0, 1.0], dtype=torch.float32)
            label_idx = torch.tensor(1, dtype=torch.long)
        else:
            label_one_hot = torch.tensor([1.0, 0.0], dtype=torch.float32)
            label_idx = torch.tensor(0, dtype=torch.long)

        # Mask generation
        if not is_fake:
            mask = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)
            source_image = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            has_source = False
        elif meta["forgery_type"] == "EFS":
            mask = torch.ones((self.image_size, self.image_size), dtype=torch.float32)
            source_image = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            has_source = False
        else:
            # AM or FS: generate synthetic source image and compute mask
            source_image = torch.rand((3, self.image_size, self.image_size), generator=g, dtype=torch.float32)
            has_source = True
            mask = generate_ground_truth_mask(
                is_fake=True,
                forgery_type=meta["forgery_type"],
                fake_image=image,
                source_image=source_image,
                image_size=self.image_size,
                threshold=0.1,
            )

        # Prompt generation
        prompts = self.prompt_generator.generate_prompts(
            is_fake=is_fake,
            forgery_type=meta["forgery_type"] if is_fake else None,
            generator=meta["generator"] if is_fake else None,
            family=meta["family"] if is_fake else None,
        )

        # Synthetic token IDs [308] within [0, s-1]
        token_ids = torch.randint(
            low=1,
            high=self.vocab_size - 1,
            size=(self.max_text_tokens,),
            generator=g,
            dtype=torch.long,
        )

        return {
            "image": image,
            "label": label_one_hot,
            "label_idx": label_idx,
            "class_target": label_idx,
            "mask": mask,
            "mask_target": mask.long(),
            "prompt": prompts["full_prompt"],
            "prompts_dict": prompts,
            "tokens": token_ids,
            "token_ids": token_ids,
            "generator": meta["generator"],
            "forgery_type": meta["forgery_type"],
            "source_image": source_image,
            "has_source": has_source,
        }

