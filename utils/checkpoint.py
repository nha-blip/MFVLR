"""Checkpoint saving and loading utilities."""

import os
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim


def create_checkpoint_state(
    model: nn.Module,
    loss_fn: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Package complete training state dictionary for checkpointing."""
    state = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
    }
    if loss_fn is not None:
        state["loss_fn_state_dict"] = loss_fn.state_dict()
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if metrics is not None:
        state["metrics"] = metrics
    if config is not None:
        state["config"] = config
    return state


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
) -> str:
    """Save model training checkpoint.

    # ASSUMPTION_FROM_PAPER_GAP:
    # Checkpoint structure and naming conventions are not specified by paper.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(state, best_filepath)
    return filepath


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    loss_fn: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model training checkpoint.

    Args:
        filepath: Path to checkpoint file.
        model: PyTorch model instance to load weights into.
        loss_fn: Optional loss function instance (e.g. MFVLRLoss) to restore state (including CMC log_tau).
        optimizer: Optional optimizer to restore state.
        scheduler: Optional learning rate scheduler to restore state.
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device)

    # Handle state dict key prefixes if any
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    if loss_fn is not None and "loss_fn_state_dict" in checkpoint:
        loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint

