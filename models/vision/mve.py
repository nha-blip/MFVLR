"""Multi-domain Vision Encoder (MVE) combining Image Encoder and shared Residual Encoder.

PAPER_SPECIFIED:
- Contains Image Encoder (IE) and Residual Encoder (RE).
- IE extracts I_loc in R^(1024 x 14 x 14) and I_g in R^(1 x 512) / R^(512) from I in R^(3 x 224 x 224).
- RE extracts I_rg in R^(1 x 512) / R^(512) from residual image I_r in R^(3 x 224 x 224).
- RE shares the exact SAME architecture and the exact SAME weights as IE.
- Fuses image and residual global features by addition:
    I_v = I_g + I_rg in R^(512)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from models.vision.image_encoder import ImageEncoder
from models.vision.residual_encoder import ResidualEncoder


class MultiDomainVisionEncoder(nn.Module):
    """Multi-domain Vision Encoder (MVE) for appearance and residual forgery encoding."""

    def __init__(
        self,
        in_channels: int = 3,
        local_channels: int = 1024,
        local_height: int = 14,
        local_width: int = 14,
        embed_dim: int = 512,
        image_transformer_blocks: int = 4,
        image_attention_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
        unet_channels: Sequence[int] = (128, 256, 512, 1024),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.local_channels = local_channels

        # 1. Single Image Encoder instance (IE)
        self.image_encoder = ImageEncoder(
            in_channels=in_channels,
            local_channels=local_channels,
            local_height=local_height,
            local_width=local_width,
            embed_dim=embed_dim,
            num_blocks=image_transformer_blocks,
            num_heads=image_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            unet_channels=unet_channels,
        )

        # 2. Residual Encoder (RE) wrapping the exact same ImageEncoder instance
        # CRITICAL: Shares the exact same weights as IE
        self.residual_encoder = ResidualEncoder(self.image_encoder)

    @property
    def ie(self) -> ImageEncoder:
        """Alias for Image Encoder."""
        return self.image_encoder

    @property
    def re(self) -> ImageEncoder:
        """Alias for Residual Encoder (direct reference to the shared ImageEncoder)."""
        return self.image_encoder

    def encode_image(
        self,
        image: torch.Tensor,
        return_skips: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        """Encode appearance image I -> (I_loc, I_g, [optional skips]).

        Args:
            image: Appearance image I of shape [B, 3, 224, 224]
            return_skips: Whether to return skip features for U-Net decoders.

        Returns:
            I_loc: [B, 1024, 14, 14]
            I_g: [B, 512]
            skips (optional): List of multi-scale tensors
        """
        return self.image_encoder(image, return_skips=return_skips)

    def encode_residual(self, i_r: torch.Tensor) -> torch.Tensor:
        """Encode residual image I_r -> I_rg via shared IE/RE.

        Args:
            i_r: Residual image tensor [B, 3, 224, 224]

        Returns:
            I_rg: Global residual feature [B, 512]
        """
        return self.residual_encoder(i_r)

    def fuse(self, i_g: torch.Tensor, i_rg: torch.Tensor) -> torch.Tensor:
        """Fuse global appearance and residual features by elementwise addition.

        # PAPER_SPECIFIED:
        # I_v = I_rg + I_g (Section III-B)

        Args:
            i_g: Global appearance feature [B, 512]
            i_rg: Global residual feature [B, 512]

        Returns:
            I_v: Fused visual feature [B, 512]
        """
        return i_g + i_rg

    def forward(
        self,
        image: torch.Tensor,
        residual_image: Optional[torch.Tensor] = None,
        return_skips: bool = False,
    ) -> Dict[str, Any]:
        """Multi-domain Vision Encoder forward pass.

        Args:
            image: Appearance image I [B, 3, 224, 224]
            residual_image: Optional residual image I_r [B, 3, 224, 224]
            return_skips: Whether to return encoder skip connections.

        Returns:
            Dictionary containing:
                "i_loc": Local appearance feature [B, 1024, 14, 14]
                "i_g": Global appearance feature [B, 512]
                "i_rg": Global residual feature [B, 512] (if residual_image provided)
                "i_v": Fused visual feature [B, 512] (if residual_image provided)
                "skips": Multi-scale skip features (if return_skips=True)
        """
        if return_skips:
            i_loc, i_g, skips = self.encode_image(image, return_skips=True)
        else:
            i_loc, i_g = self.encode_image(image, return_skips=False)
            skips = None

        outputs = {
            "i_loc": i_loc,
            "i_g": i_g,
        }
        if skips is not None:
            outputs["skips"] = skips

        if residual_image is not None:
            i_rg = self.encode_residual(residual_image)
            i_v = self.fuse(i_g, i_rg)
            outputs["i_rg"] = i_rg
            outputs["i_v"] = i_v

        return outputs
