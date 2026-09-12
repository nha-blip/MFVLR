"""Fine-grained Language Transformer (FLT) Module for MFVLR.

PAPER_SPECIFIED:
- Contains Language Encoder (LE, E=12 blocks) and Language Decoder (LD, D=7 blocks).
- Integrates Vision Injection Modules (VIM) inside both LE and LD.
- Inputs: Token IDs [B, 308] and fused visual feature I_v [B, 512].
- Outputs:
    T_low: Low-level token embeddings [B, 308, 512]
    T_hig: High-level contextualized language representations [B, 308, 512]
    T_l: Global language representation [B, 512] (strictly LAST token: T_hig[:, -1, :])
    T_rec: Reconstructed language representations [B, 308, 512]
    T_pre: Vocabulary logits [B, 308, 49408] via tied W_voc^T (Eq. 24)
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from models.language.embeddings import LanguageEmbeddings
from models.language.language_encoder import LanguageEncoder, LanguageEncoderOutput
from models.language.language_decoder import LanguageDecoder, LanguageDecoderOutput


class FLTOutput(dict):
    """Output container for FLT forward pass.

    Provides attribute access, dictionary key access, and tuple unpacking:
        t_low, t_hig, t_l, t_rec, t_pre = flt_output
    """

    def __init__(
        self,
        t_low: torch.Tensor,
        t_hig: torch.Tensor,
        t_l: torch.Tensor,
        t_rec: torch.Tensor,
        t_pre: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(
            t_low=t_low,
            t_hig=t_hig,
            t_l=t_l,
            t_rec=t_rec,
            t_pre=t_pre,
            **kwargs,
        )
        self.t_low = t_low
        self.t_hig = t_hig
        self.t_l = t_l
        self.t_rec = t_rec
        self.t_pre = t_pre

    def __iter__(self):
        # Enables tuple unpacking
        return iter((self.t_low, self.t_hig, self.t_l, self.t_rec, self.t_pre))


class FineGrainedLanguageTransformer(nn.Module):
    """Fine-grained Language Transformer (FLT) integrating Language Encoder and Decoder."""

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        max_text_tokens: int = 308,
        encoder_blocks: int = 12,
        decoder_blocks: int = 7,
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

        # 1. Language Encoder (E = 12 blocks)
        self.encoder = LanguageEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_text_tokens=max_text_tokens,
            num_blocks=encoder_blocks,
            num_heads=num_heads,
            vim_heads=vim_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            pos_init_std=pos_init_std,
        )

        # 2. Language Decoder (D = 7 blocks) with TRUE tied vocabulary weights W_voc
        self.decoder = LanguageDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_text_tokens=max_text_tokens,
            num_blocks=decoder_blocks,
            num_heads=num_heads,
            vim_heads=vim_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            pos_init_std=pos_init_std,
            token_embedding=self.encoder.embeddings.token_embed,
        )

    @property
    def token_embedding(self) -> nn.Embedding:
        """Shared vocabulary embedding matrix W_voc."""
        return self.encoder.embeddings.token_embed

    def encode(
        self,
        token_ids: torch.Tensor,
        i_v: torch.Tensor,
    ) -> LanguageEncoderOutput:
        """Encode token sequence with vision injection across 12 blocks."""
        return self.encoder(token_ids, i_v)

    def decode(
        self,
        t_low: torch.Tensor,
        t_hig: torch.Tensor,
        i_v: torch.Tensor,
        return_logits: bool = True,
    ) -> LanguageDecoderOutput:
        """Decode and reconstruct token sequence across 7 blocks."""
        return self.decoder(t_low, t_hig, i_v, return_logits=return_logits)

    def forward(
        self,
        token_ids: torch.Tensor,
        i_v: torch.Tensor,
        return_logits: bool = True,
    ) -> FLTOutput:
        """Execute full Fine-grained Language Transformer forward pass.

        Args:
            token_ids: LongTensor of token IDs [B, 308]
            i_v: Fused visual feature I_v [B, 512] or [B, 1, 512]
            return_logits: Whether to compute vocabulary reconstruction logits T_pre [B, 308, 49408].

        Returns:
            FLTOutput containing t_low, t_hig, t_l, t_rec, t_pre.
        """
        # 1. Encode: token_ids + I_v -> T_low, T_hig, T_l
        enc_out = self.encoder(token_ids, i_v)

        # 2. Decode: T_low + T_hig + I_v -> T_rec, T_pre
        dec_out = self.decoder(
            t_low=enc_out.t_low,
            t_hig=enc_out.t_hig,
            i_v=i_v,
            return_logits=return_logits,
        )

        return FLTOutput(
            t_low=enc_out.t_low,
            t_hig=enc_out.t_hig,
            t_l=enc_out.t_l,
            t_rec=dec_out.t_rec,
            t_pre=dec_out.t_pre,
        )


# Alias for concise reference
FLT = FineGrainedLanguageTransformer
