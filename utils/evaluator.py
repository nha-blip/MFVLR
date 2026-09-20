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
    total_intersections = [0, 0]
    total_unions = [0, 0]
    has_masks = False

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

            # 2. Predicted masks via argmax over 2-class localization logits (streaming IoU accumulation)
            if mask_target is not None:
                has_masks = True
                pred_masks = out.predict_mask().cpu().numpy()  # [B, H, W]
                true_masks = (mask_target.cpu().numpy() > 0.5).astype(np.uint8)  # [B, H, W]
                for c in range(2):
                    true_c = (true_masks == c)
                    pred_c = (pred_masks == c)
                    total_intersections[c] += int(np.logical_and(true_c, pred_c).sum())
                    total_unions[c] += int(np.logical_or(true_c, pred_c).sum())

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

    # Compute localization metrics (mIoU) from streaming accumulation
    if has_masks:
        eps = 1e-7
        ious = []
        for c in range(2):
            if total_unions[c] == 0:
                iou_c = 1.0
            else:
                iou_c = (total_intersections[c] + eps) / (total_unions[c] + eps)
            ious.append(float(iou_c))
        results["miou"] = float(np.mean(ious)) * 100.0
        results["iou_class_0"] = ious[0] * 100.0
        results["iou_class_1"] = ious[1] * 100.0

    return results
