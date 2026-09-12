"""Comprehensive verification suite for both DeepFakeFace and Celeb-DF-v2 datasets.
Verifies image loading, tensor shapes, mask generation, tokenization, forward-backward training, and evaluation.
"""

import os
import pytest
import torch
from torch.utils.data import DataLoader

from models.mfvlr import MFVLR
from models.losses import MFVLRLoss
from datasets.manifest_dataset import MFVLRDataset
from utils.trainer import create_optimizer, train_step
from utils.evaluator import evaluate


DATASETS = {
    "DeepFakeFace": {
        "root": "data/DeepFakeFace",
        "train": "data/DeepFakeFace/train_manifest.jsonl",
        "val": "data/DeepFakeFace/val_manifest.jsonl",
        "test": "data/DeepFakeFace/test_manifest.jsonl",
    },
    "Celeb-DF-v2": {
        "root": "data/Celeb-DF-v2",
        "train": "data/Celeb-DF-v2/train_manifest.jsonl",
        "val": "data/Celeb-DF-v2/val_manifest.jsonl",
        "test": "data/Celeb-DF-v2/test_manifest.jsonl",
    },
}


@pytest.mark.parametrize("ds_name,cfg", DATASETS.items())
def test_dataset_manifest_and_sample_loading(ds_name, cfg):
    """Verify manifest exists, loads non-empty, and item tensors have correct shapes."""
    train_manifest = cfg["train"]
    assert os.path.exists(train_manifest), f"Train manifest missing for {ds_name}: {train_manifest}"

    dataset = MFVLRDataset(manifest_path=train_manifest, dataset_root=cfg["root"])
    assert len(dataset) > 0, f"Dataset {ds_name} is empty!"

    # Test first 4 samples
    for i in range(min(4, len(dataset))):
        sample = dataset[i]
        assert "image" in sample
        assert sample["image"].shape == (3, 224, 224)
        assert sample["image"].dtype == torch.float32

        assert "label" in sample
        assert sample["label"].item() in (0, 1)

        assert "mask" in sample
        assert sample["mask"].shape == (224, 224)

        assert "token_ids" in sample
        assert sample["token_ids"].shape == (308,)


@pytest.mark.parametrize("ds_name,cfg", DATASETS.items())
def test_training_step_on_dataset(ds_name, cfg):
    """Verify single forward-backward training step executes cleanly on dataset batch."""
    dataset = MFVLRDataset(manifest_path=cfg["train"], dataset_root=cfg["root"])
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    model = MFVLR()
    loss_fn = MFVLRLoss()
    optimizer = create_optimizer(model, loss_fn, lr=1e-4)

    step_losses = train_step(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        batch=batch,
        device=torch.device("cpu"),
    )

    assert "total_loss" in step_losses
    assert step_losses["total_loss"] > 0.0
    assert "loss_fd" in step_losses
    assert "loss_fl" in step_losses


@pytest.mark.parametrize("ds_name,cfg", DATASETS.items())
def test_evaluation_pipeline_on_dataset(ds_name, cfg):
    """Verify image-only evaluation pipeline executes cleanly on dataset batch."""
    dataset = MFVLRDataset(manifest_path=cfg["test"], dataset_root=cfg["root"])
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_loader = [next(iter(loader))]

    model = MFVLR()
    metrics = evaluate(
        model=model,
        dataloader=batch_loader,
        device=torch.device("cpu"),
        positive_label=1,
    )

    assert "acc" in metrics
    assert "miou" in metrics
