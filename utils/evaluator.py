"""Evaluation utilities for MFVLR image-only inference.

PAPER_SPECIFIED:
- Standard inference uses image-only input: MVE + VD + Detection Head.
- No text prompts, no token IDs, no tokenizer, and no language path (LE, LD, FLT, CMC, KL) are executed.
- Evaluation metrics (Section IV-A):
    * Accuracy (ACC)
    * Area Under Receiver Operating Characteristic Curve (AUC)
    * Mean of class-wise Intersection over Union (mIoU)
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mfvlr import MFVLR, MFVLRInferenceOutput
from utils.metrics import compute_classification_metrics, compute_localization_metrics


from tqdm import tqdm


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    positive_label: int = 1,
) -> Dict[str, float]:
    """Evaluate MFVLR model on a test/validation dataloader using image-only inference.

    PAPER_SPECIFIED:
    - Strictly executes image-only path via model.forward_image_only(image).
    - Computes detection accuracy (ACC), continuous AUC, and localization mIoU.

    Args:
        model: MFVLR model instance.
        dataloader: PyTorch DataLoader supplying evaluation samples.
        device: Computation device.
        positive_label: Class index representing manipulated/fake face for AUC calculation (default: 1).

    Returns:
        Dictionary containing 'acc', 'auc', 'miou' (and per-class localization metrics).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    all_y_true: List[int] = []
    all_y_probs: List[float] = []
    all_mask_true: List[np.ndarray] = []
    all_mask_pred: List[np.ndarray] = []

    pbar = tqdm(dataloader, desc="Evaluating", dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)

            # Ground truth targets
            class_target = batch.get("label", batch.get("class_target", batch.get("y_target")))
            mask_target = batch.get("mask", batch.get("mask_target", batch.get("m_target")))

            # Execute strictly IMAGE-ONLY inference
            out: MFVLRInferenceOutput = model.forward_image_only(image)

            # 1. Detection probability of positive (fake) class for continuous AUC
            probs = torch.softmax(out.y_pre, dim=1)[:, positive_label].cpu().numpy()
            all_y_probs.extend(probs.tolist())

            if class_target is not None:
                if class_target.ndim > 1:
                    class_target = torch.argmax(class_target, dim=-1)
                all_y_true.extend(class_target.cpu().numpy().tolist())

            # 2. Predicted masks via argmax over 2-class localization logits
            pred_masks = out.predict_mask().cpu().numpy()
            all_mask_pred.append(pred_masks)

            if mask_target is not None:
                all_mask_true.append(mask_target.cpu().numpy())

    results: Dict[str, float] = {}

    # Compute classification metrics (ACC, AUC)
    if len(all_y_true) > 0 and len(all_y_probs) > 0:
        cls_metrics = compute_classification_metrics(
            y_true=np.array(all_y_true),
            y_pred_probs=np.array(all_y_probs),
            pos_label=positive_label,
        )
        results["acc"] = cls_metrics["acc"]
        results["auc"] = cls_metrics["auc"]

    # Compute localization metrics (mIoU)
    if len(all_mask_true) > 0 and len(all_mask_pred) > 0:
        stacked_mask_true = np.concatenate(all_mask_true, axis=0)
        stacked_mask_pred = np.concatenate(all_mask_pred, axis=0)
        loc_metrics = compute_localization_metrics(
            mask_true=stacked_mask_true,
            mask_pred=stacked_mask_pred,
            num_classes=2,
        )
        results["miou"] = loc_metrics["miou"]
        for k, v in loc_metrics.items():
            if k != "miou":
                results[k] = v

    return results
