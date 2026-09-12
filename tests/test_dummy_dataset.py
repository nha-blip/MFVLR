"""Test DummyMFVLRDataset and DataLoader functionality."""

import pytest
import torch
from torch.utils.data import DataLoader

from datasets.dummy_dataset import DummyMFVLRDataset


def test_dummy_dataset_length():
    """Verify dataset returns expected sample count."""
    dataset = DummyMFVLRDataset(num_samples=16)
    assert len(dataset) == 16


def test_dummy_dataset_item_structure():
    """Verify all expected keys and tensor shapes per sample."""
    dataset = DummyMFVLRDataset(num_samples=8, image_size=224, max_text_tokens=308, vocab_size=49408)
    sample = dataset[0]

    # Required fields from reproduction prompt Section 26
    assert "image" in sample
    assert "label" in sample
    assert "label_idx" in sample
    assert "mask" in sample
    assert "prompt" in sample
    assert "tokens" in sample
    assert "generator" in sample
    assert "forgery_type" in sample

    # Check tensor shapes and types
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 224, 224)
    assert sample["image"].dtype == torch.float32

    assert isinstance(sample["label"], torch.Tensor)
    assert sample["label"].shape == (2,)

    assert isinstance(sample["mask"], torch.Tensor)
    assert sample["mask"].shape == (224, 224)
    assert sample["mask"].dtype == torch.float32

    assert isinstance(sample["tokens"], torch.Tensor)
    assert sample["tokens"].shape == (308,)
    assert sample["tokens"].dtype == torch.long
    assert sample["tokens"].min() >= 0
    assert sample["tokens"].max() < 49408


def test_dummy_dataloader_batching():
    """Verify DataLoader batches tensors to [B, 3, 224, 224], [B, 224, 224], [B, 308]."""
    dataset = DummyMFVLRDataset(num_samples=16)
    batch_size = 8  # Paper specified batch size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in loader:
        assert batch["image"].shape == (8, 3, 224, 224)
        assert batch["label"].shape == (8, 2)
        assert batch["mask"].shape == (8, 224, 224)
        assert batch["tokens"].shape == (8, 308)
        assert len(batch["prompt"]) == 8
        break
