"""Shared U-Net Decoder Trunk for Vision Decoder (AD and MD).

PAPER_SPECIFIED:
- Takes local appearance feature I_loc in R^(B x 1024 x 14 x 14)
- Outputs feature map at 224 x 224 resolution for appearance reconstruction and mask prediction
- The U-Net decoder used by AD and MD must be the SAME network with the SAME weights.

ASSUMPTION_FROM_PAPER_GAP:
- 4-stage upsampling decoder topology:
    Stage 1: 14 x 14 -> 28 x 28 (1024 -> 512 channels)
    Stage 2: 28 x 28 -> 56 x 56 (512 -> 256 channels)
    Stage 3: 56 x 56 -> 112 x 112 (256 -> 128 channels)
    Stage 4: 112 x 112 -> 224 x 224 (128 -> 64 channels)
- Bilinear interpolation (or ConvTranspose2d) + ConvBlock with GroupNorm (32 groups) and GELU.
- Skip connection concatenation if skips provided from UNetEncoder.
"""

from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision.unet_encoder import ConvBlock


class UpsampleBlock(nn.Module):
    """Upsampling block: Upsample (2x) + ConvBlock with optional skip concatenation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.skip_channels = skip_channels
        self.conv = ConvBlock(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Upsample 2x and fuse with optional skip connection.

        Args:
            x: Input feature tensor [B, C_in, H, W]
            skip: Optional skip feature tensor [B, C_skip, 2H, 2W]

        Returns:
            Output feature tensor [B, C_out, 2H, 2W]
        """
        # Bilinear upsample 2x
        x_up = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

        if self.skip_channels > 0:
            if skip is not None:
                if x_up.shape[2:] != skip.shape[2:]:
                    x_up = F.interpolate(x_up, size=skip.shape[2:], mode="bilinear", align_corners=False)
                x_up = torch.cat([x_up, skip], dim=1)
            else:
                # If skips not provided, pad with zero channels
                B, _, H, W = x_up.shape
                zero_pad = torch.zeros((B, self.skip_channels, H, W), dtype=x_up.dtype, device=x_up.device)
                x_up = torch.cat([x_up, zero_pad], dim=1)

        return self.conv(x_up)


class UNetDecoderTrunk(nn.Module):
    """Shared U-Net Decoder Trunk converting I_loc (14x14) to 224x224 feature map.

    CRITICAL REQUIREMENT (PAPER_SPECIFIED):
    AD and MD share the exact SAME instance of this decoder trunk.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        channels: Sequence[int] = (512, 256, 128, 64),
        skip_channels: Sequence[int] = (512, 256, 128, 64),
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        s3, s2, s1, s0 = skip_channels

        # Stage 1: 14 -> 28 (1024 + s3 [512] -> c1 [512])
        self.up1 = UpsampleBlock(in_channels, c1, skip_channels=s3, num_groups=num_groups, dropout=dropout)

        # Stage 2: 28 -> 56 (c1 [512] + s2 [256] -> c2 [256])
        self.up2 = UpsampleBlock(c1, c2, skip_channels=s2, num_groups=num_groups, dropout=dropout)

        # Stage 3: 56 -> 112 (c2 [256] + s1 [128] -> c3 [128])
        self.up3 = UpsampleBlock(c2, c3, skip_channels=s1, num_groups=num_groups, dropout=dropout)

        # Stage 4: 112 -> 224 (c3 [128] + s0 [64] -> c4 [64])
        self.up4 = UpsampleBlock(c3, c4, skip_channels=s0, num_groups=num_groups, dropout=dropout)

        self.out_channels = c4

    def forward(
        self,
        i_loc: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Decode local bottleneck I_loc to 224x224 feature representation.

        Args:
            i_loc: Local feature [B, 1024, 14, 14]
            skips: Optional multi-scale encoder skip list [s0, s1, s2, s3]:
                s0: [B, 64, 224, 224]
                s1: [B, 128, 112, 112]
                s2: [B, 256, 56, 56]
                s3: [B, 512, 28, 28]

        Returns:
            Decoded feature map [B, 64, 224, 224]
        """
        if skips is not None and len(skips) >= 4:
            s0, s1, s2, s3 = skips[0], skips[1], skips[2], skips[3]
        else:
            s0 = s1 = s2 = s3 = None

        # 14 -> 28
        d1 = self.up1(i_loc, s3)
        # 28 -> 56
        d2 = self.up2(d1, s2)
        # 56 -> 112
        d3 = self.up3(d2, s1)
        # 112 -> 224
        d4 = self.up4(d3, s0)

        return d4
