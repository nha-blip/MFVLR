"""Ground truth mask generator for forgery localization.

PAPER_SPECIFIED (Section III-E, Eq. 18):
- For AM / FS:
    1. Measure absolute pixel-wise difference in RGB channels: |I_fake - I_source|
    2. Convert into grayscale
    3. Divide by 255 to yield a map in [0, 1]
    4. Apply threshold of 0.1
- Real face images: ground truth mask = 0 (all zeros, 224x224)
- Entire synthesized face images (EFS): ground truth mask = 1 (all ones, 224x224)
"""

from typing import Optional, Sequence, Union
import numpy as np
import torch
from PIL import Image


def generate_ground_truth_mask(
    is_fake: bool,
    forgery_type: Optional[str] = None,
    fake_image: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
    source_image: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
    image_size: int = 224,
    threshold: float = 0.1,
    grayscale_weights: Sequence[float] = (0.299, 0.587, 0.114),
) -> torch.Tensor:
    """Generate ground truth binary localization mask M in R^(H x W).

    # PAPER_SPECIFIED:
    # Real = zeros, EFS = ones. AM/FS = absolute RGB difference -> grayscale -> /255 -> threshold 0.1.
    #
    # ASSUMPTION_FROM_PAPER_GAP:
    # Grayscale conversion weights standard luminance (0.299, 0.587, 0.114).
    # Threshold comparison: value > threshold.

    Returns:
        torch.Tensor of shape [image_size, image_size] with dtype float32 (values in {0.0, 1.0}).
    """
    if not is_fake:
        # Real image: all zeros
        return torch.zeros((image_size, image_size), dtype=torch.float32)

    if forgery_type == "EFS":
        # Entire face synthesis: all ones
        return torch.ones((image_size, image_size), dtype=torch.float32)

    # For AM or FS, compute from fake and source images
    if forgery_type in ("AM", "FS") or forgery_type is None:
        if source_image is None or fake_image is None:
            # Fallback for un-paired datasets where source counterpart is unavailable
            return torch.ones((image_size, image_size), dtype=torch.float32)

        # Convert inputs to numpy float32 [H, W, 3] in [0, 255]
        if isinstance(fake_image, Image.Image):
            fake_arr = np.array(fake_image.resize((image_size, image_size))).astype(np.float32)
        elif isinstance(fake_image, torch.Tensor):
            fake_arr = fake_image.detach().cpu().numpy()
            if fake_arr.ndim == 3 and fake_arr.shape[0] == 3:
                fake_arr = np.transpose(fake_arr, (1, 2, 0))
            if fake_arr.max() <= 1.0:
                fake_arr = fake_arr * 255.0
        else:
            fake_arr = np.array(fake_image, dtype=np.float32)

        if isinstance(source_image, Image.Image):
            src_arr = np.array(source_image.resize((image_size, image_size))).astype(np.float32)
        elif isinstance(source_image, torch.Tensor):
            src_arr = source_image.detach().cpu().numpy()
            if src_arr.ndim == 3 and src_arr.shape[0] == 3:
                src_arr = np.transpose(src_arr, (1, 2, 0))
            if src_arr.max() <= 1.0:
                src_arr = src_arr * 255.0
        else:
            src_arr = np.array(source_image, dtype=np.float32)

        # 1. Absolute pixel-wise RGB difference
        diff = np.abs(fake_arr - src_arr)

        # 2. Grayscale conversion using luminance weights
        w_r, w_g, w_b = grayscale_weights
        gray = diff[..., 0] * w_r + diff[..., 1] * w_g + diff[..., 2] * w_b

        # 3. Divide by 255 to yield map in [0, 1]
        norm_diff = gray / 255.0

        # 4. Threshold at 0.1
        binary_mask = (norm_diff > threshold).astype(np.float32)
        return torch.from_numpy(binary_mask)

    # For any other unhandled fake type, default to ones
    return torch.ones((image_size, image_size), dtype=torch.float32)

