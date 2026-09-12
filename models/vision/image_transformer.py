"""Image Transformer and Transformer Blocks for MVE.

PAPER_SPECIFIED:
- Image Transformer block count: B = 4 (Eq. 2)
- Input sequence: I_1^tra in R^(197 x 512) (196 spatial tokens + 1 class token)
- Output representation: I_TE in R^(197 x 512)
- Global appearance feature: I_g in R^(1 x 512) is the class token of I_TE.

ASSUMPTION_FROM_PAPER_GAP:
- Pre-LayerNorm self-attention with residual connections.
- Number of attention heads: 8 (head_dim = 64).
- Feed-forward expansion dimension: 2048 (4 * 512).
- Activation: GELU.
- Dropout: 0.0.
"""

from typing import Optional
import torch
import torch.nn as nn


class ImageTransformerBlock(nn.Module):
    """Single Image Transformer Block (TB_j^i in Eq. 2)."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        act_layer = nn.GELU if activation.lower() == "gelu" else nn.ReLU
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            act_layer(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Transformer block with pre-norm residual connections.

        Args:
            x: Input token sequence [B, 197, 512]

        Returns:
            Output token sequence [B, 197, 512]
        """
        # Multi-Head Self-Attention with residual
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + attn_out

        # Feed-Forward Network with residual
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        return x


class ImageTransformer(nn.Module):
    """Image Transformer composed of B Transformer blocks.

    # PAPER_SPECIFIED:
    # B = 4 Image Transformer blocks (Eq. 2).
    """

    def __init__(
        self,
        num_blocks: int = 4,
        embed_dim: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim

        self.blocks = nn.ModuleList([
            ImageTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential execution through B blocks (Eq. 2).

        Args:
            x: Input sequence I_1^tra [B, 197, 512]

        Returns:
            I_TE: Output sequence [B, 197, 512]
        """
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
