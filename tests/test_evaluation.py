"""Tests for MFVLR evaluation pipeline and checkpoint save/load integration (Phase 10).

PAPER_SPECIFIED:
- Evaluates detection Accuracy (ACC), Area Under Curve (AUC), and localization mIoU.
- Evaluation runs strictly via image-only inference (no FLT/text path).
- Checkpointing preserves model parameters and trainable loss parameter (CMC log_tau).
"""

import math
import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.mfvlr import MFVLR
from models.losses import MFVLRLoss
from utils.trainer import create_optimizer, create_scheduler
from utils.evaluator import evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint, create_checkpoint_state


class SyntheticEvalDataset(Dataset):
    """Synthetic dataset for evaluation tests."""

    def __init__(self, size: int = 4):
        self.size = size
        torch.manual_seed(123)
        self.images = torch.randn(size, 3, 224, 224).clamp(0.0, 1.0)
        self.labels = torch.tensor([0, 1, 0, 1][:size], dtype=torch.long)
        self.masks = torch.randint(0, 2, (size, 224, 224), dtype=torch.long)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
            "mask": self.masks[idx],
        }


def test_evaluate_image_only_metrics():
    """Verify evaluate function computes ACC, continuous AUC, and mIoU without invoking FLT."""
    model = MFVLR()
    dataset = SyntheticEvalDataset(size=4)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    results = evaluate(model=model, dataloader=dataloader, device=torch.device("cpu"))

    # 1. Output keys
    assert "acc" in results
    assert "auc" in results
    assert "miou" in results

    # 2. Metric value ranges [0, 100]
    assert 0.0 <= results["acc"] <= 100.0
    assert 0.0 <= results["auc"] <= 100.0
    assert 0.0 <= results["miou"] <= 100.0


def test_evaluation_strictly_does_not_execute_flt(monkeypatch):
    """CRITICAL test: Prove evaluation loop never invokes FLT."""
    model = MFVLR()
    dataset = SyntheticEvalDataset(size=2)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    def raise_flt_error(*args, **kwargs):
        raise RuntimeError("FLT must NOT execute in evaluation loop!")

    monkeypatch.setattr(model.flt, "forward", raise_flt_error)

    results = evaluate(model=model, dataloader=dataloader, device=torch.device("cpu"))
    assert "acc" in results
    assert "auc" in results
    assert "miou" in results


def test_checkpoint_save_and_load_full_system(tmp_path):
    """Verify checkpoint save and load restores model, loss_fn (including log_tau), optimizer, and scheduler."""
    model = MFVLR()
    loss_fn = MFVLRLoss()
    optimizer = create_optimizer(model, loss_fn, lr=1e-4)
    scheduler = create_scheduler(optimizer, step_size=15, gamma=0.1)

    # Step optimizer and scheduler once
    optimizer.step()
    scheduler.step()

    # Capture initial reference weights
    initial_detection_fc = model.detection_head.fc.weight.clone().detach()
    initial_log_tau = loss_fn.cmc_loss.log_tau.clone().detach()
    initial_lr = optimizer.param_groups[0]["lr"]

    # Package and save checkpoint state
    state = create_checkpoint_state(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        step=10,
        metrics={"acc": 95.0, "auc": 98.0, "miou": 85.0},
    )
    checkpoint_dir = str(tmp_path / "checkpoints")
    saved_path = save_checkpoint(state, checkpoint_dir=checkpoint_dir, filename="test_ckpt.pt", is_best=True)

    assert os.path.exists(saved_path)
    assert os.path.exists(os.path.join(checkpoint_dir, "best_model.pt"))

    # Mutate model weights, log_tau, and optimizer
    with torch.no_grad():
        model.detection_head.fc.weight.add_(1.0)
        loss_fn.cmc_loss.log_tau.add_(0.5)

    assert not torch.equal(model.detection_head.fc.weight, initial_detection_fc)
    assert not torch.equal(loss_fn.cmc_loss.log_tau, initial_log_tau)

    # Fresh optimizer and scheduler
    new_optimizer = create_optimizer(model, loss_fn, lr=5e-4)
    new_scheduler = create_scheduler(new_optimizer, step_size=15, gamma=0.1)

    # Load checkpoint
    loaded_ckpt = load_checkpoint(
        filepath=saved_path,
        model=model,
        loss_fn=loss_fn,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        device=torch.device("cpu"),
    )

    # Verify complete restoration
    assert loaded_ckpt["epoch"] == 1
    assert loaded_ckpt["step"] == 10
    assert torch.equal(model.detection_head.fc.weight, initial_detection_fc)
    assert torch.equal(loss_fn.cmc_loss.log_tau, initial_log_tau)
    assert math.isclose(new_optimizer.param_groups[0]["lr"], initial_lr, rel_tol=1e-6)
