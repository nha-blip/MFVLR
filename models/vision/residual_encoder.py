"""Residual Encoder (RE) interface for MFVLR.

CRITICAL REQUIREMENT (PAPER_SPECIFIED):
- RE shares the exact SAME architecture and the exact SAME weights as Image Encoder (IE).
- RE is NOT an independently instantiated module or deep copy.
- This module provides a convenience wrapper ensuring parameter identity with IE.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn

from models.vision.image_encoder import ImageEncoder


class ResidualEncoder(nn.Module):
    """Residual Encoder wrapper around a shared ImageEncoder instance.

    Ensures RE strictly uses the shared weights of IE for residual encoding.
    """

    def __init__(self, shared_image_encoder: ImageEncoder):
        super().__init__()
        # Store direct reference to the shared ImageEncoder instance
        self.image_encoder = shared_image_encoder

    def forward(self, i_r: torch.Tensor) -> torch.Tensor:
        """Encode residual image I_r to global residual feature I_rg.

        Args:
            i_r: Residual image tensor [B, 3, 224, 224] (I_r = |I_pre - I|)

        Returns:
            I_rg: Global residual feature [B, 512]
        """
        # Run shared ImageEncoder on residual image
        _, i_rg = self.image_encoder(i_r, return_skips=False)
        return i_rg
