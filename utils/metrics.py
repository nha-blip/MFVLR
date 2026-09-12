"""Evaluation metrics computation for detection (ACC, AUC) and localization (mIoU).

PAPER_SPECIFIED metrics (Section IV-A):
- Accuracy (ACC)
- Area Under Receiver Operating Characteristic Curve (AUC)
- Mean of class-wise Intersection over Union (mIoU)
"""

from typing import Dict, Union
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_probs: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    pos_label: int = 1,
) -> Dict[str, float]:
    """Compute detection metrics: Accuracy (ACC) and Area Under Curve (AUC).

    Args:
        y_true: Ground truth binary labels, shape [N].
        y_pred_probs: Predicted probability of positive (fake) class, shape [N].
        threshold: Classification threshold for converting probability to class prediction.
        pos_label: Class index representing the positive/fake class (default: 1).

    Returns:
        Dictionary containing 'acc' and 'auc' scores in percentage [0, 100].
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()

    y_true = y_true.astype(int).ravel()
    y_pred_probs = y_pred_probs.ravel()

    # Map y_true to binary 1 for pos_label (fake) and 0 for negative (real)
    y_true_binary = (y_true == pos_label).astype(int)

    # Binary prediction based on threshold on positive class probability
    y_pred = (y_pred_probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true_binary, y_pred)) * 100.0

    # AUC calculation
    try:
        # Check if both classes are present in y_true_binary
        if len(np.unique(y_true_binary)) > 1:
            auc = float(roc_auc_score(y_true_binary, y_pred_probs)) * 100.0
        else:
            auc = 50.0  # Undefined when single class present
    except ValueError:
        auc = 50.0

    return {"acc": acc, "auc": auc}


def compute_localization_metrics(
    mask_true: Union[np.ndarray, torch.Tensor],
    mask_pred: Union[np.ndarray, torch.Tensor],
    num_classes: int = 2,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Compute pixel-level localization metric: mean class-wise IoU (mIoU).

    # PAPER_SPECIFIED:
    # Mean of class-wise Intersection over Union (mIoU).
    #
    # ASSUMPTION_FROM_PAPER_GAP:
    # Numerical epsilon 1e-7 for zero-division avoidance; binary thresholding via argmax.

    Args:
        mask_true: Ground-truth mask, shape [N, H, W] or [H, W], values in {0, 1}.
        mask_pred: Predicted mask, shape [N, num_classes, H, W] (logits) or [N, H, W] (class indices).
        num_classes: Number of classes (default: 2 for unmanipulated / manipulated).
        eps: Small epsilon to prevent division by zero.

    Returns:
        Dictionary containing 'miou' and per-class IoU in percentage [0, 100].
    """
    if isinstance(mask_true, torch.Tensor):
        mask_true = mask_true.detach().cpu().numpy()
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.detach().cpu().numpy()

    # If predicted mask has class dimension (e.g. logits [N, 2, H, W]), take argmax
    if mask_pred.ndim == 4 and mask_pred.shape[1] == num_classes:
        mask_pred = np.argmax(mask_pred, axis=1)

    mask_true = (mask_true > 0.5).astype(int).ravel()
    mask_pred = (mask_pred > 0.5).astype(int).ravel()

    ious = []
    for c in range(num_classes):
        true_c = (mask_true == c)
        pred_c = (mask_pred == c)
        intersection = np.logical_and(true_c, pred_c).sum()
        union = np.logical_or(true_c, pred_c).sum()
        if union == 0:
            iou_c = 1.0  # Perfect agreement on empty class
        else:
            iou_c = (intersection + eps) / (union + eps)
        ious.append(float(iou_c))

    miou = float(np.mean(ious)) * 100.0
    return {
        "miou": miou,
        "iou_class_0": ious[0] * 100.0,
        "iou_class_1": ious[1] * 100.0,
    }
