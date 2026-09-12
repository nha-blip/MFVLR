"""Test utilities: seed, metrics, checkpoint, logger, visualization."""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.seed import set_seed
from utils.metrics import compute_classification_metrics, compute_localization_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger
from utils.visualization import save_visualization, tensor_to_pil


def test_seed_determinism():
    """Verify set_seed produces reproducible tensors."""
    set_seed(42)
    t1 = torch.rand(10)
    set_seed(42)
    t2 = torch.rand(10)
    assert torch.all(t1 == t2)


def test_classification_metrics():
    """Verify ACC and AUC calculations."""
    y_true = np.array([0, 0, 1, 1])
    y_pred_probs = np.array([0.1, 0.2, 0.8, 0.9])
    res = compute_classification_metrics(y_true, y_pred_probs)

    assert res["acc"] == 100.0
    assert res["auc"] == 100.0

    # Partial accuracy
    y_pred_probs_partial = np.array([0.9, 0.2, 0.8, 0.9])
    res_partial = compute_classification_metrics(y_true, y_pred_probs_partial)
    assert res_partial["acc"] == 75.0


def test_localization_metrics():
    """Verify mIoU calculation for 2 classes."""
    mask_true = np.zeros((10, 10))
    mask_true[0:5, 0:5] = 1.0

    mask_pred = np.zeros((10, 10))
    mask_pred[0:5, 0:5] = 1.0

    res = compute_localization_metrics(mask_true, mask_pred)
    assert res["miou"] == 100.0
    assert res["iou_class_0"] == 100.0
    assert res["iou_class_1"] == 100.0


def test_checkpoint_save_and_load():
    """Verify saving and loading model weights."""
    temp_dir = tempfile.mkdtemp()
    try:
        model = nn.Linear(512, 2)
        initial_weight = model.weight.clone()

        state = {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
        }
        ckpt_path = save_checkpoint(state, temp_dir, "test.pt")
        assert os.path.exists(ckpt_path)

        # Mutate model weights
        with torch.no_grad():
            model.weight.fill_(0.0)
        assert not torch.all(model.weight == initial_weight)

        # Restore from checkpoint
        load_checkpoint(ckpt_path, model)
        assert torch.all(model.weight == initial_weight)
    finally:
        shutil.rmtree(temp_dir)


def test_visualization_save():
    """Verify multi-panel visualization exports properly."""
    temp_dir = tempfile.mkdtemp()
    try:
        img = torch.rand((3, 224, 224))
        recon = torch.rand((3, 224, 224))
        res = torch.abs(recon - img)
        gt_mask = torch.zeros((224, 224))
        pred_mask = torch.zeros((2, 224, 224))

        out_file = os.path.join(temp_dir, "test_vis.png")
        save_visualization(img, recon, res, gt_mask, pred_mask, save_path=out_file)
        assert os.path.exists(out_file)
        assert os.path.getsize(out_file) > 0
    finally:
        shutil.rmtree(temp_dir)
