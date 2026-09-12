"""Tests for MFVLR manifest-driven dataset, mask generation, and tokenizer contract (Phase 11).

PAPER_SPECIFIED:
- Image shape [3, 224, 224], float [0, 1]
- Localization mask [224, 224] in {0, 1}
- Token IDs sequence length 308, values in [0, 49407]
- Real mask = 0 (all zeros), EFS mask = 1 (all ones)
- AM/FS mask derived from |I_fake - I_source| > 0.1
- Missing source image for AM/FS must raise a clear validation error.
"""

import json
import os
import pytest
from PIL import Image
import torch

from datasets.manifest_dataset import MFVLRDataset
from datasets.tokenizer import MFVLRTokenizer, tokenize
from datasets.mask_generator import generate_ground_truth_mask


@pytest.fixture
def temp_dataset_env(tmp_path):
    """Create a temporary dataset environment with synthetic images and manifest."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create real image
    real_img_path = img_dir / "real_01.jpg"
    Image.new("RGB", (224, 224), color=(100, 150, 200)).save(real_img_path)

    # 2. Create EFS image
    efs_img_path = img_dir / "efs_01.jpg"
    Image.new("RGB", (224, 224), color=(200, 100, 50)).save(efs_img_path)

    # 3. Create AM fake and source images
    am_source_path = img_dir / "am_source_01.jpg"
    Image.new("RGB", (224, 224), color=(128, 128, 128)).save(am_source_path)

    am_fake_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    # Draw a 50x50 modified patch with large difference (> 0.1 threshold)
    for x in range(50, 100):
        for y in range(50, 100):
            am_fake_img.putpixel((x, y), (255, 255, 255))
    am_fake_path = img_dir / "am_fake_01.jpg"
    am_fake_img.save(am_fake_path)

    # 4. Create sample manifest records
    manifest_records = [
        {
            "image_path": str(real_img_path),
            "label": "real",
            "manipulation_type": "Real",
            "generator": "Real",
        },
        {
            "image_path": str(efs_img_path),
            "label": "fake",
            "manipulation_type": "EFS",
            "generator": "DDPM",
            "family": "diffusion",
        },
        {
            "image_path": str(am_fake_path),
            "source_image_path": str(am_source_path),
            "label": "fake",
            "manipulation_type": "AM",
            "generator": "Diffae",
            "family": "diffusion",
        },
    ]

    # Save JSONL manifest
    manifest_jsonl = tmp_path / "manifest.jsonl"
    with open(manifest_jsonl, "w", encoding="utf-8") as f:
        for r in manifest_records:
            f.write(json.dumps(r) + "\n")

    return {
        "manifest_jsonl": str(manifest_jsonl),
        "real_img_path": str(real_img_path),
        "efs_img_path": str(efs_img_path),
        "am_fake_path": str(am_fake_path),
        "am_source_path": str(am_source_path),
        "tmp_path": tmp_path,
    }


def test_manifest_dataset_loading_and_tensor_shapes(temp_dataset_env):
    """Verify manifest dataset loads correctly and yields exact tensor shapes and value ranges."""
    dataset = MFVLRDataset(
        manifest_path=temp_dataset_env["manifest_jsonl"],
        real_class_index=0,
        fake_class_index=1,
    )

    assert len(dataset) == 3

    # Sample 0: Real
    real_sample = dataset[0]
    assert real_sample["image"].shape == (3, 224, 224)
    assert real_sample["image"].dtype == torch.float32
    assert (real_sample["image"] >= 0.0).all() and (real_sample["image"] <= 1.0).all()
    assert real_sample["label"].item() == 0
    assert real_sample["mask"].shape == (224, 224)
    assert real_sample["mask"].dtype == torch.long
    assert (real_sample["mask"] == 0).all(), "Real sample mask must be all zeros"
    assert real_sample["token_ids"].shape == (308,)
    assert real_sample["token_ids"].dtype == torch.long
    assert (real_sample["token_ids"] >= 0).all() and (real_sample["token_ids"] < 49408).all()

    # Sample 1: EFS (Full synthesis)
    efs_sample = dataset[1]
    assert efs_sample["label"].item() == 1
    assert efs_sample["mask"].shape == (224, 224)
    assert (efs_sample["mask"] == 1).all(), "EFS sample mask must be all ones"

    # Sample 2: AM (Manipulated with source image)
    am_sample = dataset[2]
    assert am_sample["label"].item() == 1
    assert am_sample["mask"].shape == (224, 224)
    # The 50x50 patch must have mask value 1, rest must be 0
    assert am_sample["mask"][60, 60].item() == 1
    assert am_sample["mask"][10, 10].item() == 0


def test_missing_source_image_raises_explicit_error(temp_dataset_env):
    """Verify that manipulated samples (AM/FS) missing source image raise a clear ValueError."""
    tmp_path = temp_dataset_env["tmp_path"]
    bad_manifest_path = tmp_path / "bad_manifest.jsonl"

    # AM sample with missing source_image_path
    with open(bad_manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "image_path": temp_dataset_env["am_fake_path"],
            "label": "fake",
            "manipulation_type": "AM",
            # source_image_path intentionally missing
        }) + "\n")

    dataset = MFVLRDataset(manifest_path=str(bad_manifest_path))

    with pytest.raises(ValueError, match="Source image and fake image are strictly required"):
        _ = dataset[0]


def test_tokenizer_contract_and_vocabulary_bounds():
    """Verify tokenizer contract: exact 308 length, dtype torch.long, values in [0, 49407]."""
    tokenizer = MFVLRTokenizer(max_tokens=308, vocab_size=49408)

    prompt = "A photo of a fake face A photo of an entire synthesized face A photo generated by the diffusion-based model The source generative model of this photo is DDPM"
    tokens = tokenizer(prompt)

    # 1. Output shape and type
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape == (308,)
    assert tokens.dtype == torch.long

    # 2. Vocabulary bounds
    assert (tokens >= 0).all()
    assert (tokens < 49408).all()

    # 3. Determinism
    tokens_2 = tokenizer(prompt)
    assert torch.equal(tokens, tokens_2)

    # 4. Input type validation
    with pytest.raises(TypeError):
        tokenizer(12345)
