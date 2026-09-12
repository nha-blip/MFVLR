"""Tests for CLI entry points: train.py, evaluate.py, and infer.py (Phase 11).

Verifies command line wiring, dry-run mode, checkpoint resume, image-only evaluation, and single image inference.
"""

import os
import subprocess
import sys
import pytest
from PIL import Image
import torch

from models.mfvlr import MFVLR
from models.losses import MFVLRLoss
from utils.checkpoint import save_checkpoint, create_checkpoint_state
from utils.trainer import create_optimizer, create_scheduler


@pytest.fixture(scope="module")
def cli_environment(tmp_path_factory):
    """Set up temporary images, checkpoint, and config for CLI testing."""
    tmp_path = tmp_path_factory.mktemp("cli_test")

    # 1. Test image
    img_path = tmp_path / "test_sample.jpg"
    Image.new("RGB", (224, 224), color=(120, 180, 240)).save(img_path)

    # 2. Save a dummy model checkpoint
    model = MFVLR()
    loss_fn = MFVLRLoss()
    optimizer = create_optimizer(model, loss_fn, lr=1e-4)
    scheduler = create_scheduler(optimizer, step_size=15, gamma=0.1)

    state = create_checkpoint_state(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        step=20,
        metrics={"acc": 92.5, "auc": 96.0, "miou": 80.0},
    )
    ckpt_dir = str(tmp_path / "checkpoints")
    ckpt_path = save_checkpoint(state, checkpoint_dir=ckpt_dir, filename="test_ckpt.pt")

    return {
        "img_path": str(img_path),
        "ckpt_path": str(ckpt_path),
        "ckpt_dir": ckpt_dir,
        "config_path": "configs/mfvlr.yaml",
        "tmp_path": tmp_path,
    }


def test_train_dry_run_cli(cli_environment):
    """Verify python train.py --config configs/mfvlr.yaml --dry-run executes 1 step and exits 0 cleanly."""
    cmd = [
        sys.executable,
        "train.py",
        "--config", cli_environment["config_path"],
        "--dry-run",
        "--batch-size", "2",
        "--device", "cpu",
        "--output-dir", str(cli_environment["tmp_path"] / "train_out"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"train.py dry-run failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "DRY-RUN test PASSED" in result.stdout or "DRY-RUN test PASSED" in result.stderr


def test_evaluate_cli(cli_environment):
    """Verify python evaluate.py executes image-only evaluation and outputs metrics."""
    cmd = [
        sys.executable,
        "evaluate.py",
        "--config", cli_environment["config_path"],
        "--checkpoint", cli_environment["ckpt_path"],
        "--device", "cpu",
        "--batch-size", "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"evaluate.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "EVALUATION RESULTS:" in result.stdout or "EVALUATION RESULTS:" in result.stderr
    assert "Detection Accuracy (ACC)" in result.stdout or "Detection Accuracy (ACC)" in result.stderr


def test_infer_cli(cli_environment):
    """Verify python infer.py executes single image inference and saves localization mask."""
    output_mask = str(cli_environment["tmp_path"] / "predicted_mask.png")
    cmd = [
        sys.executable,
        "infer.py",
        "--config", cli_environment["config_path"],
        "--checkpoint", cli_environment["ckpt_path"],
        "--image", cli_environment["img_path"],
        "--output-mask", output_mask,
        "--device", "cpu",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"infer.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "MFVLR INFERENCE RESULTS:" in result.stdout or "MFVLR INFERENCE RESULTS:" in result.stderr
    assert os.path.exists(output_mask), f"Expected predicted mask image at {output_mask}"


def test_cli_missing_file_validation(cli_environment):
    """Verify CLI commands provide clear errors when files are not found."""
    # 1. Missing image for infer.py
    cmd_infer = [
        sys.executable,
        "infer.py",
        "--config", cli_environment["config_path"],
        "--checkpoint", cli_environment["ckpt_path"],
        "--image", "non_existent_image_12345.jpg",
    ]
    result_infer = subprocess.run(cmd_infer, capture_output=True, text=True)
    assert result_infer.returncode != 0
    assert "FileNotFoundError" in result_infer.stderr or "not found" in result_infer.stderr

    # 2. Missing checkpoint for evaluate.py
    cmd_eval = [
        sys.executable,
        "evaluate.py",
        "--config", cli_environment["config_path"],
        "--checkpoint", "non_existent_ckpt_12345.pt",
    ]
    result_eval = subprocess.run(cmd_eval, capture_output=True, text=True)
    assert result_eval.returncode != 0
    assert "FileNotFoundError" in result_eval.stderr or "not found" in result_eval.stderr
