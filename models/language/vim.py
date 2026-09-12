"""Vision Injection Module (VIM) for vision-guided language representation learning.

PAPER_SPECIFIED (Section III-D, Eq. 4-11):
- Inputs:
    Language token sequence T_tok^j in R^(B x n x d) = R^(B x 308 x 512)
    Visual class token feature I_v in R^(B x 1 x d) = R^(B x 1 x 512)
- Q/K/V mapping (Eq. 4-6):
    q_j = T_tok^j W_que^j  (Language = Query)
    k_j = I_v W_key^j      (Vision = Key)
    v_j = I_v W_val^j      (Vision = Value)
    where W_que^j, W_key^j, W_val^j in R^(d x d)
- Partition into r heads (Eq. 7-9):
    Q_{j,i} in R^(B x r x n x (d/r))
    K_{j,i} in R^(B x r x 1 x (d/r))
    V_{j,i} in R^(B x r x 1 x (d/r))
- Cross-attention per head (Eq. 10):
    T_{j,i}^glo = softmax( Q_{j,i} K_{j,i}^T / sqrt(d/r) ) V_{j,i}
- Concatenation & Output projection + residual addition (Eq. 11):
    T_j^glo = Cat({T_{j,i}^glo}_{i=1}^r) in R^(B x n x d)
    T_j^add = T_j^glo W_fc^j + T_tok^j in R^(B x n x d)

ASSUMPTION_FROM_PAPER_GAP:
- Head count r is unspecified; reproduction sets r = 8 (head_dim = 64).
- Linear bias is enabled by default in PyTorch nn.Linear.
- I_v of shape [B, d] is automatically unsqueezed to [B, 1, d] for flexible interface compatibility.
- Mathematical Singleton K/V property: Because K has length 1, softmax over dim=-1 is mathematically identically 1.0.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


class VisionInjectionModule(nn.Module):
    """Vision Injection Module (VIM) implementing Eq. (4)-(11) from the paper."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 512 // 8 = 64
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Trainable projection matrices (Eq. 4-6)
        # Language -> Query
        self.w_que = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Vision -> Key
        self.w_key = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Vision -> Value
        self.w_val = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Trainable output projection matrix (Eq. 11)
        self.w_fc = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        t_tok: torch.Tensor,
        i_v: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute vision-language interaction.

        Args:
            t_tok: Language representation tensor [B, n, 512] (e.g. n = 308)
            i_v: Visual global representation tensor [B, 1, 512] or [B, 512]
            return_attention: Whether to return the attention probability tensor.

        Returns:
            If return_attention is False:
                t_add: Enhanced vision-language token tensor [B, n, 512]
            If return_attention is True:
                (t_add, attention_probs)
                where attention_probs has shape [B, num_heads, n, 1]
        """
        B, n, d = t_tok.shape

        # Support flexible visual input: [B, 512] -> [B, 1, 512]
        if i_v.ndim == 2:
            i_v = i_v.unsqueeze(1)  # [B, 1, d]

        # 1. Linear Projections (Eq. 4-6)
        # Language = Query: [B, n, d]
        q = self.w_que(t_tok)
        # Vision = Key: [B, 1, d]
        k = self.w_key(i_v)
        # Vision = Value: [B, 1, d]
        v = self.w_val(i_v)

        # 2. Partition into r heads (Eq. 7-9)
        # Q: [B, r, n, head_dim]
        Q = q.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        # K: [B, r, 1, head_dim]
        K = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        # V: [B, r, 1, head_dim]
        V = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Cross-Attention calculation (Eq. 10)
        # Raw attention scores: [B, r, n, head_dim] @ [B, r, head_dim, 1] -> [B, r, n, 1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Softmax over key dimension (dim=-1, which has length 1)
        # Note: Mathematically, softmax over a singleton dimension evaluates identically to 1.0.
        attn_probs = torch.softmax(scores, dim=-1)  # [B, r, n, 1]

        # Attention applied to Value: [B, r, n, 1] @ [B, r, 1, head_dim] -> [B, r, n, head_dim]
        t_glo_heads = torch.matmul(attn_probs, V)  # [B, r, n, head_dim]

        # 4. Head concatenation: [B, n, r * head_dim] = [B, n, d] (Eq. 11)
        t_glo = t_glo_heads.transpose(1, 2).contiguous().view(B, n, d)

        # 5. Output projection and residual addition (Eq. 11)
        # T_add = T_glo W_fc + T_tok
        t_add = self.w_fc(t_glo) + t_tok  # [B, n, d]

        if return_attention:
            return t_add, attn_probs
        return t_add


# Alias for concise reference
VIM = VisionInjectionModule
