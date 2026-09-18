"""Training orchestration utilities for MFVLR.

PAPER_SPECIFIED:
- Optimizer: Adam (not AdamW)
- Learning rate: 1e-4
- Weight decay: 1e-3
- Learning rate schedule: divided by 10 every 15 epochs (StepLR with step_size=15, gamma=0.1)
- Loss parameter: CMC trainable temperature tau (log_tau) is optimized along with model parameters.
- Multi-task loss: L = L_fd + L_lr + L_cmc + L_fl + L_ar + L_kl (Eq. 27).

ASSUMPTION_FROM_PAPER_GAP:
- AMP (Automatic Mixed Precision) is optional and disabled by default (enabled=False).
- Device is configurable (CPU/CUDA via torch.device).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from models.mfvlr import MFVLR, MFVLROutput
from models.losses import MFVLRLoss, TotalLossOutput


def create_optimizer(
    model: nn.Module,
    loss_fn: Optional[nn.Module] = None,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> optim.Adam:
    """Create paper-specified Adam optimizer over model and loss parameters.

    PAPER_SPECIFIED:
    - Optimizer: Adam
    - Learning rate: 1e-4
    - Weight decay: 1e-3
    - Loss parameters (such as CMC log_tau) are included in the parameter list.

    Args:
        model: MFVLR model instance.
        loss_fn: Optional MFVLRLoss instance containing trainable loss parameters (CMC log_tau).
        lr: Learning rate (default: 1e-4).
        weight_decay: Weight decay factor (default: 1e-3).
        betas: Adam beta coefficients (default: (0.9, 0.999)).
        eps: Adam numerical epsilon (default: 1e-8).

    Returns:
        torch.optim.Adam optimizer instance.
    """
    # Collect unique trainable parameters without duplicates
    unique_params: set = set()

    for p in model.parameters():
        if p.requires_grad:
            unique_params.add(p)

    if loss_fn is not None:
        for p in loss_fn.parameters():
            if p.requires_grad:
                unique_params.add(p)

    param_list = list(unique_params)

    optimizer = optim.Adam(
        param_list,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    step_size: int = 15,
    gamma: float = 0.1,
    last_epoch: int = -1,
) -> StepLR:
    """Create paper-specified StepLR learning rate scheduler.

    PAPER_SPECIFIED:
    - Learning rate is divided by 10 every 15 epochs.
    - Equivalent to StepLR with step_size=15 and gamma=0.1.

    Args:
        optimizer: Optimizer instance to wrap.
        step_size: Period of learning rate decay in epochs (default: 15).
        gamma: Multiplicative factor of learning rate decay (default: 0.1).
        last_epoch: Index of last epoch for resumption (default: -1).

    Returns:
        torch.optim.lr_scheduler.StepLR instance.
    """
    return StepLR(
        optimizer=optimizer,
        step_size=step_size,
        gamma=gamma,
        last_epoch=last_epoch,
    )


def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    batch: Dict[str, Any],
    device: Optional[torch.device] = None,
    scaler: Optional[Any] = None,
    use_amp: bool = False,
    precision: str = "fp32",
    return_outputs: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor, torch.Tensor]]:
    """Execute a single MFVLR training step with forward, loss computation, backward, and optimizer update.

    Args:
        model: MFVLR model instance.
        loss_fn: MFVLRLoss loss function instance.
        optimizer: Adam optimizer instance.
        batch: Batch dictionary from DataLoader containing:
            - 'image': Tensor [B, 3, 224, 224]
            - 'token_ids': Tensor [B, 308]
            - 'label' / 'class_target': Tensor [B]
            - 'mask' / 'mask_target': Tensor [B, 224, 224]
        device: Target computation device (default: device of model parameters).
        scaler: Optional GradScaler for AMP FP16.
        use_amp: Whether to use Automatic Mixed Precision (default: False).
        precision: Precision mode ('bf16' / 'bfloat16', 'fp16' / 'float16', or 'fp32').
        return_outputs: Whether to return predicted and ground truth labels.

    Returns:
        Dictionary of detached scalar loss values.
    """
    if device is None:
        device = next(model.parameters()).device

    # Extract inputs and targets from batch
    image = batch["image"].to(device, non_blocking=True)
    raw_tokens = batch.get("token_ids", batch.get("tokens"))
    if raw_tokens is None:
        raise KeyError("Batch dictionary must contain 'token_ids' or 'tokens' key.")
    token_ids = raw_tokens.to(device, non_blocking=True)

    # Support multiple common target key naming conventions
    raw_label = batch.get("label", batch.get("class_target", batch.get("y_target", batch.get("label_idx"))))
    if raw_label is None:
        raise KeyError("Batch dictionary must contain 'label' or 'class_target' key.")
    class_target = raw_label.to(device, non_blocking=True)
    if class_target.ndim > 1:
        class_target = torch.argmax(class_target, dim=-1)

    raw_mask = batch.get("mask", batch.get("mask_target", batch.get("m_target")))
    if raw_mask is None:
        raise KeyError("Batch dictionary must contain 'mask' or 'mask_target' key.")
    mask_target = raw_mask.to(device, non_blocking=True)
    if mask_target.dtype != torch.long:
        mask_target = mask_target.long()

    model.train()
    optimizer.zero_grad()

    # Collect all trainable parameters from model and loss_fn (e.g. trainable CMC tau)
    all_trainable_params = [
        p for p in list(model.parameters()) + list(loss_fn.parameters())
        if p.requires_grad
    ]

    is_cuda = (device.type == "cuda")
    norm_precision = precision.lower().strip()
    is_bf16 = norm_precision in ("bf16", "bfloat16") and is_cuda
    is_fp16 = norm_precision in ("fp16", "float16", "amp") and is_cuda

    if is_bf16:
        # BFloat16 mode (recommended on modern GPUs like L40, A100, RTX 30/40)
        # Prevents FP16 overflow/underflow without requiring GradScaler
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs: MFVLROutput = model(image=image, token_ids=token_ids, return_logits=True)
            losses: TotalLossOutput = loss_fn(
                y_pre=outputs.y_pre,
                y_target=class_target,
                t_pre=outputs.t_pre,
                target_token_ids=token_ids,
                i_v=outputs.i_v,
                t_l=outputs.t_l,
                m_pre=outputs.m_pre,
                target_mask=mask_target,
                i_pre=outputs.i_pre,
                image=image,
                t_lpre=outputs.t_lpre,
            )
        losses.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
        optimizer.step()

    elif is_fp16 and scaler is not None:
        # FP16 mode with GradScaler
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs: MFVLROutput = model(image=image, token_ids=token_ids, return_logits=True)
            losses: TotalLossOutput = loss_fn(
                y_pre=outputs.y_pre,
                y_target=class_target,
                t_pre=outputs.t_pre,
                target_token_ids=token_ids,
                i_v=outputs.i_v,
                t_l=outputs.t_l,
                m_pre=outputs.m_pre,
                target_mask=mask_target,
                i_pre=outputs.i_pre,
                image=image,
                t_lpre=outputs.t_lpre,
            )
        scaler.scale(losses.total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    else:
        # Full precision (FP32)
        outputs: MFVLROutput = model(image=image, token_ids=token_ids, return_logits=True)
        losses: TotalLossOutput = loss_fn(
            y_pre=outputs.y_pre,
            y_target=class_target,
            t_pre=outputs.t_pre,
            target_token_ids=token_ids,
            i_v=outputs.i_v,
            t_l=outputs.t_l,
            m_pre=outputs.m_pre,
            target_mask=mask_target,
            i_pre=outputs.i_pre,
            image=image,
            t_lpre=outputs.t_lpre,
        )
        losses.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
        optimizer.step()

    loss_dict = {
        "total_loss": float(losses.total_loss.detach().item()),
        "loss_fd": float(losses.loss_fd.detach().item()),
        "loss_lr": float(losses.loss_lr.detach().item()),
        "loss_cmc": float(losses.loss_cmc.detach().item()),
        "loss_fl": float(losses.loss_fl.detach().item()),
        "loss_ar": float(losses.loss_ar.detach().item()),
        "loss_kl": float(losses.loss_kl.detach().item()),
    }

    if return_outputs:
        return loss_dict, outputs.y_pre.detach(), class_target.detach()

    return loss_dict


from tqdm import tqdm
from sklearn.metrics import average_precision_score


def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: Optional[torch.device] = None,
    scaler: Optional[Any] = None,
    use_amp: bool = False,
    precision: str = "fp32",
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    logger: Optional[Any] = None,
) -> Dict[str, float]:
    """Train MFVLR model for one complete epoch over a DataLoader with per-batch ACC, AP, and loss logging.

    Args:
        model: MFVLR model instance.
        loss_fn: MFVLRLoss instance.
        dataloader: PyTorch DataLoader supplying training batches.
        optimizer: Adam optimizer.
        device: Computation device.
        scaler: Optional GradScaler for AMP.
        use_amp: Whether to use AMP (default: False).
        epoch: Current epoch index (0-indexed).
        total_epochs: Total number of epochs.
        logger: Optional logger instance.

    Returns:
        Dictionary of mean scalar losses and classification metrics over the epoch.
    """
    model.train()
    total_batches = len(dataloader)
    if total_batches == 0:
        return {
            "total_loss": 0.0,
            "loss_fd": 0.0,
            "loss_lr": 0.0,
            "loss_cmc": 0.0,
            "loss_fl": 0.0,
            "loss_ar": 0.0,
            "loss_kl": 0.0,
            "acc": 0.0,
            "ap": 0.0,
        }

    running_losses: Dict[str, float] = {
        "total_loss": 0.0,
        "loss_fd": 0.0,
        "loss_lr": 0.0,
        "loss_cmc": 0.0,
        "loss_fl": 0.0,
        "loss_ar": 0.0,
        "loss_kl": 0.0,
    }

    all_y_true: List[int] = []
    all_y_probs: List[float] = []
    running_correct: int = 0
    running_samples: int = 0

    desc = f"Epoch {epoch + 1}/{total_epochs}" if epoch is not None and total_epochs is not None else "Training"
    pbar = tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=True)

    for batch_idx, batch in enumerate(pbar):
        step_losses, y_pre, y_true = train_step(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch=batch,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            precision=precision,
            return_outputs=True,
        )
        for key, val in step_losses.items():
            running_losses[key] += val

        # Compute real-time batch & running Accuracy and Average Precision (AP)
        with torch.no_grad():
            preds = torch.argmax(y_pre, dim=1)
            probs = torch.softmax(y_pre, dim=1)[:, 1] if y_pre.shape[1] > 1 else torch.sigmoid(y_pre[:, 0])

            batch_correct = int((preds == y_true).sum().item())
            running_correct += batch_correct
            running_samples += int(y_true.numel())

            all_y_true.extend(y_true.cpu().tolist())
            all_y_probs.extend(probs.cpu().tolist())

            running_acc = (running_correct / running_samples) * 100.0

            # Compute running Average Precision (AP)
            running_ap = running_acc
            if len(set(all_y_true)) > 1:
                try:
                    running_ap = float(average_precision_score(all_y_true, all_y_probs) * 100.0)
                except Exception:
                    running_ap = running_acc

        avg_loss = running_losses["total_loss"] / (batch_idx + 1)
        pbar.set_postfix({
            "loss": f"{step_losses['total_loss']:.2f}",
            "acc": f"{running_acc:.1f}%",
            "ap": f"{running_ap:.1f}%",
            "fd": f"{step_losses['loss_fd']:.2f}",
            "fl": f"{step_losses['loss_fl']:.2f}",
        })

    mean_losses = {key: val / total_batches for key, val in running_losses.items()}
    mean_losses["acc"] = float(running_acc)
    mean_losses["ap"] = float(running_ap)
    return mean_losses
