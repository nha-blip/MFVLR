"""Image Encoder (IE) for MFVLR.

PAPER_SPECIFIED:
- Input image I in R^(3 x 224 x 224)
- Extracts I_loc in R^(1024 x 14 x 14) via U-Net encoder
- Flattens spatial dimensions (14 x 14 = 196) and linearly projects 1024 -> 512
- Appends learnable class token: I_tok in R^(197 x 512) (Eq. 1)
- Adds learnable positional embedding P_i: I_1^tra = I_tok + P_i in R^(197 x 512)
- Passes through B = 4 Image Transformer blocks: I_TE in R^(197 x 512) (Eq. 2)
- Extracts global appearance forgery representation I_g in R^(1 x 512) / R^(512) from class token of I_TE.

ASSUMPTION_FROM_PAPER_GAP:
- Class token and positional embedding initialized with truncated normal (std = 0.02).
- Class token placed at index 0 (standard convention).
- U-Net encoder channel schedule [128, 256, 512, 1024].
- 8 attention heads, 2048 feed-forward width, GELU activation.
"""

from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from models.vision.unet_encoder import UNetEncoder
from models.vision.image_transformer import ImageTransformer


class ImageEncoder(nn.Module):
    """Image Encoder (IE) extracting local feature I_loc and global appearance feature I_g."""

    def __init__(
        self,
        in_channels: int = 3,
        local_channels: int = 1024,
        local_height: int = 14,
        local_width: int = 14,
        embed_dim: int = 512,
        num_blocks: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
        unet_channels: Sequence[int] = (128, 256, 512, 1024),
    ):
        super().__init__()
        self.local_channels = local_channels
        self.local_height = local_height
        self.local_width = local_width
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_spatial_tokens = local_height * local_width  # 14 * 14 = 196
        self.total_tokens = self.num_spatial_tokens + 1       # 196 + 1 = 197

        # 1. U-Net Encoder: 3 x 224 x 224 -> 1024 x 14 x 14
        self.unet_encoder = UNetEncoder(
            in_channels=in_channels,
            channels=unet_channels,
            dropout=dropout,
        )

        # 2. Linear projection: 1024 -> 512 (Proj in Eq. 1)
        self.proj = nn.Linear(local_channels, embed_dim)

        # 3. Learnable class token (App in Eq. 1)
        # ASSUMPTION_FROM_PAPER_GAP: Initialized with truncated normal std=0.02
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 4. Learnable positional embedding P_i (Eq. 1)
        # ASSUMPTION_FROM_PAPER_GAP: Initialized with truncated normal std=0.02
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 5. B = 4 Image Transformer blocks (Eq. 2)
        self.transformer = ImageTransformer(
            num_blocks=num_blocks,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

    def extract_local(
        self,
        x: torch.Tensor,
        return_skips: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Run U-Net encoder to extract I_loc in R^(B x 1024 x 14 x 14)."""
        return self.unet_encoder(x, return_skips=return_skips)

    def extract_global_from_local(
        self,
        i_loc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert I_loc to tokens, add class token & pos embed, run Transformer.

        Args:
            i_loc: Local feature tensor [B, 1024, 14, 14]

        Returns:
            Tuple of:
                i_g: Global feature extracted from class token [B, 512]
                i_te: Full output token sequence [B, 197, 512]
        """
        B, C, H, W = i_loc.shape
        # Flatten spatial dimensions: [B, 1024, 14, 14] -> [B, 196, 1024]
        # (Flat in Eq. 1)
        flat_loc = i_loc.flatten(2).transpose(1, 2)

        # Project to embed_dim: [B, 196, 1024] -> [B, 196, 512]
        # (Proj in Eq. 1)
        projected = self.proj(flat_loc)

        # Append learnable class token: [B, 1, 512] + [B, 196, 512] -> [B, 197, 512]
        # (App in Eq. 1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        i_tok = torch.cat([cls_tokens, projected], dim=1)  # [B, 197, 512]

        # Add learnable positional embedding P_i (Eq. 1)
        i_1_tra = i_tok + self.pos_embed  # [B, 197, 512]

        # Sequential execution through B = 4 Image Transformer blocks (Eq. 2)
        i_te = self.transformer(i_1_tra)  # [B, 197, 512]

        # Global representation I_g from class token (index 0)
        i_g = i_te[:, 0, :]  # [B, 512]
        return i_g, i_te

    def forward(
        self,
        x: torch.Tensor,
        return_skips: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        """Full Image Encoder forward pass.

        Args:
            x: Input image [B, 3, 224, 224]
            return_skips: Whether to return U-Net encoder skip features.

        Returns:
            If return_skips is False:
                (I_loc [B, 1024, 14, 14], I_g [B, 512])
            If return_skips is True:
                (I_loc [B, 1024, 14, 14], I_g [B, 512], skips)
        """
        if return_skips:
            i_loc, skips = self.unet_encoder(x, return_skips=True)
            i_g, _ = self.extract_global_from_local(i_loc)
            return i_loc, i_g, skips
        else:
            i_loc = self.unet_encoder(x, return_skips=False)
            i_g, _ = self.extract_global_from_local(i_loc)
            return i_loc, i_g
