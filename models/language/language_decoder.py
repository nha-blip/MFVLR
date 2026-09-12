"""Language Decoder (LD) for Fine-grained Language Reconstruction.

PAPER_SPECIFIED:
- Vocabulary size: s = 49,408
- Maximum text tokens: n = 308
- Embedding dimension: d = 512
- Language Decoder depth: D = 7 blocks (Section III-D, Eq. 12)
- Shifted low-level language input (Section III-D):
    Prepend begin token (BOS) and remove final token from T_low^e:
    T_shift = [t_BOS, t_1, t_2, ..., t_307] in R^(B x 308 x 512)
    Initial decoder state: T_t1^tra = T_shift + P_d in R^(B x 308 x 512)
- Sequential block flow (Section III-D, Eq. 13-16):
    1. Masked Multi-Head Self-Attention with causal mask and residual:
       T_tj^mmha = MMHA_j^d(LN(T_tj^tra)) + T_tj^tra
    2. Encoder-Decoder Multi-Head Attention (cross-attention) with residual:
       T_tj^mha = MHA_j^d(LN(T_tj^mmha), T_hig^e) + T_tj^mmha
    3. Feed-Forward Network with residual:
       T_tj^ff = FF_j^d(LN(T_tj^mha)) + T_tj^mha
    4. Vision Injection Module (VIM) with residual:
       T_t(j+1)^tra = VIM_j^d(T_tj^ff, I_v)  (VIM includes internal residual)
- Reconstructed representation after D=7 blocks: T_rec^d in R^(B x 308 x 512) (Eq. 12, no final LayerNorm)
- Vocabulary reconstruction projection (Eq. 24):
    T_pre = T_rec^d W_voc^T in R^(B x 308 x 49408)
    Uses TRUE weight tying with Language Encoder vocabulary embedding matrix W_voc.

ASSUMPTION_FROM_PAPER_GAP:
- Pre-LayerNorm Transformer block architecture.
- Learnable BOS embedding initialized with truncated normal (std = 0.02).
- Learnable decoder positional embedding P_d initialized with truncated normal (std = 0.02).
- Attention heads: 8 (head_dim = 64).
- Feed-forward expansion dimension: 2048 (4 * 512).
- Activation: GELU.
- Dropout: 0.0.
- Each of the 7 LD blocks instantiates an independent VIM module.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.language.vim import VisionInjectionModule


class LanguageDecoderOutput(dict):
    """Output container for Language Decoder forward pass.

    Provides attribute access, dictionary key access, and tuple unpacking:
        t_rec, t_pre = ld_output
    """

    def __init__(
        self,
        t_rec: torch.Tensor,
        t_pre: Optional[torch.Tensor] = None,
        t_t1_tra: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(t_rec=t_rec, t_pre=t_pre, **kwargs)
        if t_t1_tra is not None:
            self["t_t1_tra"] = t_t1_tra
        self.t_rec = t_rec
        self.t_pre = t_pre
        self.t_t1_tra = t_t1_tra

    def __iter__(self):
        # Enables tuple unpacking: t_rec, t_pre = decoder(...)
        return iter((self.t_rec, self.t_pre))


class LanguageDecoderBlock(nn.Module):
    """Single Language Decoder block with causal MMHA, cross-attention, FFN, and VIM (Eq. 13-16)."""

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

        # 1. Masked Multi-Head Self-Attention with Pre-LN (MMHA_j^d, Eq. 13)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 2. Encoder-Decoder Multi-Head Attention with Pre-LN (MHA_j^d, Eq. 14)
        # Query = Decoder state, Key = T_hig^e, Value = T_hig^e
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 3. Feed-Forward Network with Pre-LN (FF_j^d, Eq. 15)
        self.norm3 = nn.LayerNorm(embed_dim)
        act_layer = nn.GELU if activation.lower() == "gelu" else nn.ReLU
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            act_layer(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # 4. Vision Injection Module (VIM_j^d, Eq. 16)
        # Distinct instance per block
        self.vim = VisionInjectionModule(
            embed_dim=embed_dim,
            num_heads=vim_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        t_hig: torch.Tensor,
        i_v: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute one Language Decoder block with literal paper ordering (Eq. 13-16).

        Flow (Section III-D):
            T_tra_tj
               ↓
            LN1
               ↓
            MMHA_j^d (causal) + Residual -> T_tj^mmha
               ↓
            LN2
               ↓
            MHA_j^d(Q=T_tj^mmha, K=T_hig, V=T_hig) + Residual -> T_tj^mha
               ↓
            LN3
               ↓
            FF_j^d + Residual -> T_tj^ff
               ↓
            VIM_j^d(T_tj^ff, I_v) -> T_tra_t(j+1)

        Args:
            x: Input decoder sequence [B, 308, 512]
            t_hig: High-level language features from LE [B, 308, 512]
            i_v: Visual global feature [B, 512] or [B, 1, 512]
            causal_mask: Boolean/float causal mask [308, 308]

        Returns:
            Output decoder sequence [B, 308, 512]
        """
        # Step 1: Causal Masked Self-Attention (Eq. 13)
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
        )
        x = x + attn_out  # T_tj^mmha [B, 308, 512]

        # Step 2: Encoder-Decoder Cross-Attention using complete T_hig^e (Eq. 14)
        norm_x = self.norm2(x)
        cross_out, _ = self.cross_attn(
            query=norm_x,
            key=t_hig,
            value=t_hig,
        )
        x = x + cross_out  # T_tj^mha [B, 308, 512]

        # Step 3: FFN with pre-LN and residual (Eq. 15)
        norm_x = self.norm3(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out  # T_tj^ff [B, 308, 512]

        # Step 4: Vision Injection Module (Eq. 16)
        # Note: VIM internally performs T_glo W_fc + T_tok (residual addition)
        x = self.vim(x, i_v)  # T_tra_t(j+1) [B, 308, 512]

        return x


class LanguageDecoder(nn.Module):
    """Language Decoder (LD) composed of D=7 blocks with causal MMHA, cross-attention, FFN, and VIM."""

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        max_text_tokens: int = 308,
        num_blocks: int = 7,
        num_heads: int = 8,
        vim_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
        pos_init_std: float = 0.02,
        token_embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_text_tokens = max_text_tokens
        self.num_blocks = num_blocks

        # 1. Begin-token (BOS) embedding: [1, 1, 512]
        # ASSUMPTION_FROM_PAPER_GAP: Learnable BOS embedding initialized with std = 0.02
        self.bos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.bos_embed, std=pos_init_std)

        # 2. Decoder positional embedding P_d: [1, 308, 512]
        # ASSUMPTION_FROM_PAPER_GAP: Learnable P_d initialized with std = 0.02
        self.pos_embed = nn.Parameter(torch.zeros(1, max_text_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=pos_init_std)

        # 3. Exactly D = 7 Language Decoder blocks (Eq. 12)
        # Each block instantiates its own distinct VIM instance
        self.blocks = nn.ModuleList([
            LanguageDecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                vim_heads=vim_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_blocks)
        ])

        # 4. Vocabulary embedding matrix W_voc for true weight tying (Eq. 24)
        # If token_embedding is provided, share its parameter; otherwise create one.
        if token_embedding is not None:
            self.token_embedding = token_embedding
        else:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

    def set_token_embedding(self, token_embedding: nn.Embedding) -> None:
        """Tie decoder vocabulary projection to encoder token embedding (W_voc)."""
        self.token_embedding = token_embedding

    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate boolean causal attention mask [seq_len, seq_len].

        True indicates masked-out positions (future tokens cannot be attended).
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def prepare_decoder_input(self, t_low: torch.Tensor) -> torch.Tensor:
        """Construct shifted decoder input sequence T_t1^tra (Section III-D).

        1. Prepend begin-token embedding (t_BOS).
        2. Remove the last token from T_low^e.
        3. Sequence length remains n = 308.
        4. Add decoder positional embedding P_d.

        Args:
            t_low: Low-level token embeddings T_low^e [B, 308, 512]

        Returns:
            t_t1_tra: Positioned shifted decoder input [B, 308, 512]
        """
        B, n, d = t_low.shape
        # Prepend BOS and drop last token: [B, 1, 512] + [B, 307, 512] -> [B, 308, 512]
        bos = self.bos_embed.expand(B, 1, d)
        shifted = torch.cat([bos, t_low[:, :-1, :]], dim=1)  # [B, 308, 512]

        # Add decoder positional embedding P_d
        t_t1_tra = shifted + self.pos_embed[:, :n, :]  # [B, 308, 512]
        return t_t1_tra

    def project_to_vocabulary(self, t_rec: torch.Tensor) -> torch.Tensor:
        """Project reconstructed representation to vocabulary logits (Eq. 24).

        T_pre = T_rec^d @ W_voc^T in R^(B x 308 x 49408)

        Uses TRUE weight tying with self.token_embedding.weight.
        """
        return F.linear(t_rec, self.token_embedding.weight)

    def forward(
        self,
        t_low: torch.Tensor,
        t_hig: torch.Tensor,
        i_v: torch.Tensor,
        return_logits: bool = True,
    ) -> LanguageDecoderOutput:
        """Reconstruct prompt text sequence across D=7 decoder blocks.

        Args:
            t_low: Low-level token embeddings T_low^e [B, 308, 512]
            t_hig: High-level contextualized representations from LE [B, 308, 512]
            i_v: Visual global feature [B, 512] or [B, 1, 512]
            return_logits: Whether to compute vocabulary logits T_pre [B, 308, 49408].

        Returns:
            LanguageDecoderOutput containing:
                t_rec: Reconstructed text representation T_rec^d [B, 308, 512]
                t_pre: Projected vocabulary logits T_pre [B, 308, 49408] (if return_logits=True)
                t_t1_tra: Initial shifted decoder representation [B, 308, 512]
        """
        # 1. Prepare shifted input: T_t1^tra [B, 308, 512]
        t_t1_tra = self.prepare_decoder_input(t_low)

        # 2. Construct causal mask
        seq_len = t_t1_tra.size(1)
        causal_mask = self.get_causal_mask(seq_len, t_t1_tra.device)

        # 3. Sequential execution through D=7 Transformer decoder blocks (Eq. 12)
        x = t_t1_tra
        for block in self.blocks:
            x = block(x, t_hig, i_v, causal_mask=causal_mask)

        # 4. PAPER_SPECIFIED (Eq. 12): T_rec^d is the direct output of block 7 (no final LayerNorm)
        t_rec = x  # [B, 308, 512]

        # 5. Vocabulary projection via tied W_voc^T (Eq. 24)
        t_pre = None
        if return_logits:
            t_pre = self.project_to_vocabulary(t_rec)  # [B, 308, 49408]

        return LanguageDecoderOutput(
            t_rec=t_rec,
            t_pre=t_pre,
            t_t1_tra=t_t1_tra,
        )
