"""Language Encoder (LE) for Multi-domain Fine-grained Vision-Language Reconstruction.

PAPER_SPECIFIED:
- Vocabulary size: s = 49,408
- Maximum text tokens: n = 308
- Embedding dimension: d = 512
- Language Encoder depth: E = 12 blocks (Section III-D, Eq. 3)
- Sequential block flow (Section III-D):
    1. Multi-head self-attention with residual:
       T_tok^j = MHA_j^e(LN_j^e(T_tra_j)) + T_tra_j
    2. Vision injection module:
       T_add^j = VIM_j(T_tok^j, I_v)
    3. Feed-forward network with residual:
       T_tra_(j+1) = FF_j^e(LN(T_add^j)) + T_add^j
- High-level representation after 12 blocks: T_hig^e in R^(B x 308 x 512)
- Global language representation: T_l in R^(B x 512) is the LAST TOKEN of T_hig^e:
    T_l = T_hig[:, -1, :]

ASSUMPTION_FROM_PAPER_GAP:
- Pre-LayerNorm Transformer block architecture.
- Attention heads: 8 (head_dim = 64).
- Feed-forward expansion dimension: 2048 (4 * 512).
- Activation: GELU.
- Dropout: 0.0.
- Each of the 12 LE blocks instantiates an independent VIM module (no weight sharing across depth).
"""

from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import torch
import torch.nn as nn

from models.language.embeddings import LanguageEmbeddings
from models.language.vim import VisionInjectionModule


class LanguageEncoderOutput(dict):
    """Output container for Language Encoder forward pass.
    
    Provides attribute access, dictionary key access, and tuple unpacking:
        t_low, t_hig, t_l = le_output
    """

    def __init__(
        self,
        t_low: torch.Tensor,
        t_hig: torch.Tensor,
        t_l: torch.Tensor,
        t_1_tra: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(t_low=t_low, t_hig=t_hig, t_l=t_l, **kwargs)
        if t_1_tra is not None:
            self["t_1_tra"] = t_1_tra
        self.t_low = t_low
        self.t_hig = t_hig
        self.t_l = t_l
        self.t_1_tra = t_1_tra

    def __iter__(self):
        # Enables tuple unpacking: t_low, t_hig, t_l = encoder(token_ids, i_v)
        return iter((self.t_low, self.t_hig, self.t_l))


class LanguageEncoderBlock(nn.Module):
    """Single Language Encoder block with MHA, VIM, and FFN (Section III-D, Eq. 3-11)."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        vim_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.vim_heads = vim_heads

        # 1. Multi-Head Self-Attention with Pre-LN (MHA_j^e)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 2. Vision Injection Module (VIM_j)
        # Distinct instance per block
        self.vim = VisionInjectionModule(
            embed_dim=embed_dim,
            num_heads=vim_heads,
        )

        # 3. Feed-Forward Network with Pre-LN (FF_j^e)
        self.norm2 = nn.LayerNorm(embed_dim)
        act_layer = nn.GELU if activation.lower() == "gelu" else nn.ReLU
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            act_layer(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self,
        t_tra: torch.Tensor,
        i_v: torch.Tensor,
    ) -> torch.Tensor:
        """Execute one Language Encoder block with vision injection.

        Flow (Section III-D):
            T_tra_j
               ↓
            LN_j^e
               ↓
            MHA_j^e + Residual -> T_tok^j
               ↓
            VIM_j(T_tok^j, I_v) -> T_add^j
               ↓
            LN
               ↓
            FF_j^e + Residual  -> T_tra_(j+1)

        Args:
            t_tra: Input language representation [B, 308, 512]
            i_v: Visual global feature [B, 512] or [B, 1, 512]

        Returns:
            t_next: Output language representation [B, 308, 512]
        """
        # Step 1: Self-Attention with pre-LN and residual
        norm_t = self.norm1(t_tra)
        attn_out, _ = self.self_attn(norm_t, norm_t, norm_t)
        t_tok = t_tra + attn_out  # [B, 308, 512]

        # Step 2: Vision injection via VIM
        t_add = self.vim(t_tok, i_v)  # [B, 308, 512]

        # Step 3: FFN with pre-LN and residual
        norm_t_add = self.norm2(t_add)
        ffn_out = self.ffn(norm_t_add)
        t_next = t_add + ffn_out  # [B, 308, 512]

        return t_next


class LanguageEncoder(nn.Module):
    """Language Encoder (LE) composed of E=12 blocks with VIM injection."""

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        max_text_tokens: int = 308,
        num_blocks: int = 12,
        num_heads: int = 8,
        vim_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
        pos_init_std: float = 0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_text_tokens = max_text_tokens
        self.num_blocks = num_blocks

        # Token embedding interface and learnable positional embeddings
        self.embeddings = LanguageEmbeddings(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_text_tokens=max_text_tokens,
            pos_init_std=pos_init_std,
        )

        # Exactly E = 12 Language Encoder blocks
        # Each block instantiates its own distinct VIM instance
        self.blocks = nn.ModuleList([
            LanguageEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                vim_heads=vim_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        token_ids: torch.Tensor,
        i_v: torch.Tensor,
    ) -> LanguageEncoderOutput:
        """Encode token sequence with vision guidance across 12 blocks.

        Args:
            token_ids: LongTensor of token IDs [B, 308]
            i_v: Visual global feature [B, 512] or [B, 1, 512]

        Returns:
            LanguageEncoderOutput containing:
                t_low: Low-level token embeddings T_low^e [B, 308, 512]
                t_hig: High-level contextualized language representation T_hig^e [B, 308, 512]
                t_l: Global language representation T_l [B, 512] (strictly LAST token: t_hig[:, -1, :])
                t_1_tra: Positioned initial token representation T_1^tra [B, 308, 512]
        """
        # 1. Embeddings: T_low^e [B, 308, 512] and T_1^tra = T_low^e + P_e [B, 308, 512]
        t_low, t_1_tra = self.embeddings(token_ids)

        # 2. Sequential execution through E=12 Transformer blocks with VIM (Eq. 3)
        x = t_1_tra
        for block in self.blocks:
            x = block(x, i_v)

        # 3. PAPER_SPECIFIED (Eq. 3): T_hig^e is the direct output of the 12th LE block (no final LayerNorm)
        t_hig = x

        # 4. PAPER_SPECIFIED: T_l is the LAST TOKEN of T_hig^e (Section III-D)
        # T_l = T_hig[:, -1, :] in R^(B x 512)
        t_l = t_hig[:, -1, :]

        return LanguageEncoderOutput(
            t_low=t_low,
            t_hig=t_hig,
            t_l=t_l,
            t_1_tra=t_1_tra,
        )
