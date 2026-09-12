"""Tests for MFVLR training step, optimizer, LR scheduler, and epoch orchestration (Phase 10).

PAPER_SPECIFIED:
- Optimizer: Adam (lr=1e-4, weight_decay=1e-3)
- Scheduler: Divide by 10 every 15 epochs (StepLR step_size=15, gamma=0.1)
- Loss parameter: Trainable CMC temperature log_tau is optimized
- Training step: Multi-task loss backward and optimizer update
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.mfvlr import MFVLR
from models.losses import MFVLRLoss
from utils.trainer import create_optimizer, create_scheduler, train_step, train_one_epoch


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def dummy_batch(batch_size):
    """Synthetic training batch dictionary."""
    torch.manual_seed(42)
    return {
        "image": torch.randn(batch_size, 3, 224, 224).clamp(0.0, 1.0),
        "token_ids": torch.randint(0, 49408, (batch_size, 308), dtype=torch.long),
        "class_target": torch.tensor([0, 1], dtype=torch.long)[:batch_size],
        "mask_target": torch.randint(0, 2, (batch_size, 224, 224), dtype=torch.long),
    }


def test_optimizer_creation_and_hyperparameters():
    """PAPER_SPECIFIED test: Verify optimizer is Adam with exact paper hyperparameters (lr=1e-4, weight_decay=1e-3)."""
    model = MFVLR()
    loss_fn = MFVLRLoss()

    optimizer = create_optimizer(model, loss_fn, lr=1e-4, weight_decay=1e-3)

    # 1. Optimizer class must be Adam, NOT AdamW
    assert isinstance(optimizer, optim.Adam)
    assert not isinstance(optimizer, optim.AdamW)

    # 2. Hyperparameters
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == 1e-4
    assert group["weight_decay"] == 1e-3

    # 3. Model parameters and loss parameter (log_tau) included without duplicates
    opt_params = set(group["params"])
    expected_unique = set(p for p in model.parameters() if p.requires_grad) | set(p for p in loss_fn.parameters() if p.requires_grad)

    assert len(group["params"]) == len(expected_unique)
    assert opt_params == expected_unique
    assert loss_fn.cmc_loss.log_tau in opt_params


def test_learning_rate_scheduler_boundary_behavior():
    """PAPER_SPECIFIED test: Verify LR divides by 10 every 15 epochs."""
    model = MFVLR()
    optimizer = create_optimizer(model, lr=1e-4)
    scheduler = create_scheduler(optimizer, step_size=15, gamma=0.1)

    assert scheduler.step_size == 15
    assert scheduler.gamma == 0.1

    # Epochs 0-14: lr == 1e-4
    for epoch in range(15):
        current_lr = optimizer.param_groups[0]["lr"]
        assert math.isclose(current_lr, 1e-4, rel_tol=1e-6), f"Epoch {epoch} expected 1e-4, got {current_lr}"
        optimizer.step()
        scheduler.step()

    # Epoch 15 (after 15th step): lr == 1e-5
    current_lr = optimizer.param_groups[0]["lr"]
    assert math.isclose(current_lr, 1e-5, rel_tol=1e-6), f"Epoch 15 expected 1e-5, got {current_lr}"

    # Epochs 16-29: lr == 1e-5
    for epoch in range(15, 29):
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        assert math.isclose(current_lr, 1e-5, rel_tol=1e-6), f"Epoch {epoch} expected 1e-5, got {current_lr}"

    # Epoch 30 (after 30th step): lr == 1e-6
    optimizer.step()
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    assert math.isclose(current_lr, 1e-6, rel_tol=1e-6), f"Epoch 30 expected 1e-6, got {current_lr}"


def test_train_step_execution_and_parameter_updates(dummy_batch):
    """Verify single training step executes forward, backward, optimizer update, and updates CMC log_tau."""
    model = MFVLR()
    loss_fn = MFVLRLoss()
    optimizer = create_optimizer(model, loss_fn, lr=1e-4, weight_decay=1e-3)

    # Record initial parameter values for checking updates
    initial_fc_weight = model.detection_head.fc.weight.clone().detach()
    initial_log_tau = loss_fn.cmc_loss.log_tau.clone().detach()

    # Execute training step
    losses = train_step(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        batch=dummy_batch,
        device=torch.device("cpu"),
    )

    # 1. Returned dictionary contains all 7 scalar losses
    expected_keys = {"total_loss", "loss_fd", "loss_lr", "loss_cmc", "loss_fl", "loss_ar", "loss_kl"}
    assert set(losses.keys()) == expected_keys
    for k, v in losses.items():
        assert isinstance(v, float)
        assert math.isfinite(v)

    # 2. Parameters have been updated by Adam
    updated_fc_weight = model.detection_head.fc.weight
    assert not torch.equal(initial_fc_weight, updated_fc_weight)

    # 3. CMC log_tau has been updated by Adam (for batch_size=2, CMC has cross-modal negatives)
    updated_log_tau = loss_fn.cmc_loss.log_tau
    assert not torch.equal(initial_log_tau, updated_log_tau)


def test_train_one_epoch_orchestration():
    """Verify train_one_epoch iterates through a DataLoader and averages losses across batches."""
    model = MFVLR()
    loss_fn = MFVLRLoss()
    optimizer = create_optimizer(model, loss_fn, lr=1e-4)

    # Construct minimal synthetic dataset with 2 batches of size 1
    class DummyTrainDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 224, 224).clamp(0.0, 1.0),
                "token_ids": torch.randint(0, 49408, (308,), dtype=torch.long),
                "label": torch.tensor(idx % 2, dtype=torch.long),
                "mask": torch.randint(0, 2, (224, 224), dtype=torch.long),
            }

    dataloader = DataLoader(DummyTrainDataset(), batch_size=1, shuffle=False)

    mean_losses = train_one_epoch(
        model=model,
        loss_fn=loss_fn,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    expected_keys = {"total_loss", "loss_fd", "loss_lr", "loss_cmc", "loss_fl", "loss_ar", "loss_kl", "acc", "ap"}
    assert set(mean_losses.keys()) == expected_keys
    for k, v in mean_losses.items():
        assert isinstance(v, float)
        assert math.isfinite(v)
