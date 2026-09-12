"""Test ground truth mask generation."""

import numpy as np
import pytest
import torch
from datasets.mask_generator import generate_ground_truth_mask


def test_real_mask_all_zeros():
    """Verify real images yield all-zero mask."""
    mask = generate_ground_truth_mask(is_fake=False, image_size=224)
    assert mask.shape == (224, 224)
    assert torch.all(mask == 0.0)


def test_efs_mask_all_ones():
    """Verify entire face synthesis yields all-one mask."""
    mask = generate_ground_truth_mask(is_fake=True, forgery_type="EFS", image_size=224)
    assert mask.shape == (224, 224)
    assert torch.all(mask == 1.0)


def test_am_fs_mask_thresholding():
    """Verify absolute RGB difference -> grayscale -> /255 -> threshold 0.1 pipeline."""
    # Synthetic fake and source image
    fake = np.zeros((224, 224, 3), dtype=np.float32)
    source = np.zeros((224, 224, 3), dtype=np.float32)

    # Set region with difference > 0.1 (e.g. diff = 50 / 255 ≈ 0.196)
    fake[50:100, 50:100, :] = 50.0

    # Set region with difference < 0.1 (e.g. diff = 10 / 255 ≈ 0.039)
    fake[150:180, 150:180, :] = 10.0

    mask = generate_ground_truth_mask(
        is_fake=True,
        forgery_type="AM",
        fake_image=fake,
        source_image=source,
        image_size=224,
        threshold=0.1,
    )

    assert mask.shape == (224, 224)
    # The [50:100, 50:100] region should be 1.0
    assert torch.all(mask[50:100, 50:100] == 1.0)
    # The [150:180, 150:180] region should be 0.0
    assert torch.all(mask[150:180, 150:180] == 0.0)
    # Background should be 0.0
    assert torch.all(mask[0:40, 0:40] == 0.0)
