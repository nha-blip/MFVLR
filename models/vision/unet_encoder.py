"""U-Net Encoder for local appearance forgery feature extraction.

PAPER_SPECIFIED:
- Input image I in R^(3 x 224 x 224)
- Local appearance feature I_loc in R^(1024 x 14 x 14) (c = 1024, h = w = 14)
- Output feature spatial stride = 16 (224 / 14 = 16)

ASSUMPTION_FROM_PAPER_GAP:
- U-Net encoder topology is not specified in the paper.
- Reproduction uses a 4-stage residual downsampling trunk:
    Stage 1: 3 -> 128 (224 x 224 -> 112 x 112)
    Stage 2: 128 -> 256 (112 x 112 -> 56 x 56)
    Stage 3: 256 -> 512 (56 x 56 -> 28 x 28)
    Stage 4: 512 -> 1024 (28 x 28 -> 14 x 14)
- Normalization: GroupNorm (32 groups).
- Activation: GELU.
- Skip connections are cached for use by shared U-Net decoders in Phase 4.
"""

from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two-convolution residual block with GroupNorm and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Ensure num_groups divides out_channels
        groups = min(num_groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.GELU()

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(groups, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = self.act2(out + res)
        return out


class DownsampleBlock(nn.Module):
    """Downsampling stage: MaxPool2d + ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, num_groups, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pooled = self.pool(x)
        return self.conv(x_pooled)


class UNetEncoder(nn.Module):
    """U-Net Encoder extracting multi-scale skip features and local bottleneck I_loc.

    # PAPER_SPECIFIED:
    # Output I_loc in R^(1024 x 14 x 14) for input in R^(3 x 224 x 224).
    #
    # ASSUMPTION_FROM_PAPER_GAP:
    # 4 stages with channel schedule [128, 256, 512, 1024].
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (128, 256, 512, 1024),
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels

        # Stage 0: Initial convolution at full resolution (224 x 224)
        self.init_conv = ConvBlock(in_channels, c1 // 2, num_groups, dropout)

        # Stage 1: 224 -> 112 (c1)
        self.down1 = DownsampleBlock(c1 // 2, c1, num_groups, dropout)

        # Stage 2: 112 -> 56 (c2)
        self.down2 = DownsampleBlock(c1, c2, num_groups, dropout)

        # Stage 3: 56 -> 28 (c3)
        self.down3 = DownsampleBlock(c2, c3, num_groups, dropout)

        # Stage 4: 28 -> 14 (c4 = 1024) -> I_loc
        self.down4 = DownsampleBlock(c3, c4, num_groups, dropout)

        self.out_channels = c4

    def forward(
        self,
        x: torch.Tensor,
        return_skips: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Extract I_loc and optionally multi-scale skip features.

        Args:
            x: Input image tensor of shape [B, 3, 224, 224]
            return_skips: Whether to return intermediate skip features for U-Net decoders.

        Returns:
            If return_skips is False:
                I_loc of shape [B, 1024, 14, 14]
            If return_skips is True:
                (I_loc, [s0, s1, s2, s3])
                where:
                    s0: [B, 64, 224, 224]
                    s1: [B, 128, 112, 112]
                    s2: [B, 256, 56, 56]
                    s3: [B, 512, 28, 28]
                    I_loc: [B, 1024, 14, 14]
        """
        s0 = self.init_conv(x)      # [B, 64, 224, 224]
        s1 = self.down1(s0)         # [B, 128, 112, 112]
        s2 = self.down2(s1)         # [B, 256, 56, 56]
        s3 = self.down3(s2)         # [B, 512, 28, 28]
        i_loc = self.down4(s3)      # [B, 1024, 14, 14]

        if return_skips:
            return i_loc, [s0, s1, s2, s3]
        return i_loc
