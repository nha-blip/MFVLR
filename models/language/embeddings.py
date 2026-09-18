"""Language Token and Positional Embedding Module for MFVLR.

PAPER_SPECIFIED:
- Vocabulary size: s = 49,408
- Maximum text tokens: n = 308
- Embedding dimension: d = 512
- Token embeddings: T_low^e in R^(B x 308 x 512) via vocabulary matrix W_voc in R^(49408 x 512)
- Positional embeddings: P_e in R^(1 x 308 x 512)
- Initial Transformer input: T_1^tra = T_low^e + P_e in R^(B x 308 x 512) (Eq. 3 context)

ASSUMPTION_FROM_PAPER_GAP:
- nn.Embedding(49408, 512) is used as the clean token embedding interface.
- Positional embeddings P_e initialized with truncated normal (std = 0.02).
- Tokenizer variant/checkpoint is an assumption (standard CLIP vocabulary size 49,408).
"""

from typing import Tuple
import torch
import torch.nn as nn


class LanguageEmbeddings(nn.Module):
    """Token and positional embedding layer for language sequences."""

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        max_text_tokens: int = 308,
        pos_init_std: float = 0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_text_tokens = max_text_tokens

        # Vocabulary embedding matrix W_voc in R^(s x d)
        # ASSUMPTION_FROM_PAPER_GAP: Initialize token embedding with std=0.02 (standard CLIP/Transformer)
        # to prevent vocabulary logits explosion and loss_lr divergence.
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.trunc_normal_(self.token_embed.weight, std=pos_init_std)

        # Learnable positional embedding P_e in R^(1 x n x d)
        # ASSUMPTION_FROM_PAPER_GAP: Truncated normal initialization
        self.pos_embed = nn.Parameter(torch.zeros(1, max_text_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=pos_init_std)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert token IDs into low-level embeddings and initial positioned representations.

        Args:
            token_ids: LongTensor of shape [B, n] (e.g. [B, 308]) with IDs in [0, s-1]

        Returns:
            Tuple of:
                t_low: Low-level token embeddings T_low^e [B, 308, 512] (preserved for LD)
                t_1_tra: Positioned token embeddings T_1^tra = T_low^e + P_e [B, 308, 512]
        """
        # T_low^e = token_embed(token_ids) in R^(B x n x d)
        t_low = self.token_embed(token_ids)

        # T_1^tra = T_low^e + P_e in R^(B x n x d)
        t_1_tra = t_low + self.pos_embed

        return t_low, t_1_tra
