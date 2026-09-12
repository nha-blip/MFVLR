"""Visualization utilities for appearance reconstruction, residuals, and localization masks."""

import os
from typing import Optional, Union
import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: Union[torch.Tensor, np.ndarray]) -> Image.Image:
    """Convert a [C, H, W] or [H, W] float tensor in [0, 1] to a PIL Image."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
        # [C, H, W] -> [H, W, C]
        tensor = np.transpose(tensor, (1, 2, 0))
        if tensor.shape[2] == 1:
            tensor = tensor.squeeze(2)

    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)


def save_visualization(
    image: Union[torch.Tensor, np.ndarray],
    reconstructed: Optional[Union[torch.Tensor, np.ndarray]] = None,
    residual: Optional[Union[torch.Tensor, np.ndarray]] = None,
    gt_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    pred_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: str = "visualizations/sample.png",
) -> None:
    """Save multi-panel visualization of image, reconstruction, residual, and masks.

    # ASSUMPTION_FROM_PAPER_GAP:
    # Visualization layout and export format.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    panels = []
    panels.append(tensor_to_pil(image))

    if reconstructed is not None:
        panels.append(tensor_to_pil(reconstructed))

    if residual is not None:
        panels.append(tensor_to_pil(residual))

    if gt_mask is not None:
        panels.append(tensor_to_pil(gt_mask))

    if pred_mask is not None:
        if isinstance(pred_mask, torch.Tensor) and pred_mask.ndim == 3 and pred_mask.shape[0] == 2:
            pred_mask = torch.argmax(pred_mask, dim=0).float()
        elif isinstance(pred_mask, np.ndarray) and pred_mask.ndim == 3 and pred_mask.shape[0] == 2:
            pred_mask = np.argmax(pred_mask, axis=0).astype(float)
        panels.append(tensor_to_pil(pred_mask))

    # Stitch horizontally
    widths, heights = zip(*(p.size for p in panels))
    total_width = sum(widths)
    max_height = max(heights)

    combined = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for p in panels:
        if p.mode != "RGB":
            p = p.convert("RGB")
        combined.paste(p, (x_offset, 0))
        x_offset += p.width

    combined.save(save_path)
