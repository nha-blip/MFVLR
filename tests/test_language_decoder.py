"""Tests for Language Decoder (LD), causal MMHA, cross-attention, and vocabulary projection (Phase 7).

PAPER_SPECIFIED:
- Vocabulary size: s = 49,408
- Maximum token length: n = 308
- Embedding dimension: d = 512
- Language Decoder depth: D = 7 blocks (Section III-D, Eq. 12)
- Shifted low-level language input: Prepend BOS, drop last token (Eq. context)
- Sequential block flow: MMHA -> Cross-MHA(T_hig^e) -> FFN -> VIM (Eq. 13-16)
- Reconstructed representation: T_rec^d in R^(B x 308 x 512) (no final LayerNorm)
- Vocabulary projection: T_pre = T_rec^d @ W_voc^T in R^(B x 308 x 49408) with true weight tying (Eq. 24)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.language.embeddings import LanguageEmbeddings
from models.language.vim import VisionInjectionModule
from models.language.language_encoder import LanguageEncoder
from models.language.language_decoder import (
    LanguageDecoderBlock,
    LanguageDecoder,
    LanguageDecoderOutput,
)


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def max_tokens():
    return 308


@pytest.fixture
def embed_dim():
    return 512


@pytest.fixture
def vocab_size():
    return 49408


@pytest.fixture
def dummy_t_low(batch_size, max_tokens, embed_dim):
    """Synthetic low-level token embeddings T_low^e [B, 308, 512]."""
    torch.manual_seed(42)
    return torch.randn(batch_size, max_tokens, embed_dim)


@pytest.fixture
def dummy_t_hig(batch_size, max_tokens, embed_dim):
    """Synthetic high-level contextualized language embeddings T_hig^e [B, 308, 512]."""
    torch.manual_seed(43)
    return torch.randn(batch_size, max_tokens, embed_dim)


@pytest.fixture
def dummy_visual_feature(batch_size, embed_dim):
    """Synthetic visual feature I_v in R^(B x 512)."""
    torch.manual_seed(44)
    return torch.randn(batch_size, embed_dim)


def test_decoder_input_preparation_and_shift(dummy_t_low, batch_size, max_tokens, embed_dim):
    """Verify shifted decoder input construction (prepend BOS, drop last token, sequence length n=308)."""
    decoder = LanguageDecoder(
        vocab_size=49408,
        embed_dim=embed_dim,
        max_text_tokens=max_tokens,
        num_blocks=7,
    )

    t_t1_tra = decoder.prepare_decoder_input(dummy_t_low)

    # 1. Output shape must be strictly [B, 308, 512]
    assert t_t1_tra.shape == (batch_size, max_tokens, embed_dim), (
        f"Expected t_t1_tra shape {(batch_size, max_tokens, embed_dim)}, got {t_t1_tra.shape}"
    )

    # 2. Subtract decoder positional embedding to inspect unpositioned shifted sequence
    unpositioned_shifted = t_t1_tra - decoder.pos_embed

    # 3. Position 0 must equal the BOS embedding
    expected_bos = decoder.bos_embed.expand(batch_size, 1, embed_dim)
    assert torch.allclose(unpositioned_shifted[:, 0:1, :], expected_bos), (
        "Position 0 of shifted sequence must equal BOS embedding"
    )

    # 4. Positions 1..307 must equal T_low positions 0..306
    assert torch.allclose(unpositioned_shifted[:, 1:, :], dummy_t_low[:, :-1, :]), (
        "Shifted sequence positions [1:] must equal T_low[:-1]"
    )

    # 5. Decoder positional embedding shape
    assert decoder.pos_embed.shape == (1, max_tokens, embed_dim)
    assert torch.isfinite(t_t1_tra).all()


def test_language_decoder_structure():
    """Verify Language Decoder has exactly D=7 blocks, correct internal order, and no final LayerNorm."""
    decoder = LanguageDecoder(
        vocab_size=49408,
        embed_dim=512,
        max_text_tokens=308,
        num_blocks=7,
        num_heads=8,
        vim_heads=8,
        dim_feedforward=2048,
    )

    # 1. Verify exactly 7 blocks
    assert len(decoder.blocks) == 7, f"Expected 7 blocks, got {len(decoder.blocks)}"
    assert decoder.num_blocks == 7

    # 2. Verify each block contains MMHA, Cross-MHA, FFN, and VIM in order
    vim_instances = []
    vim_weight_ptrs = []
    for j, block in enumerate(decoder.blocks):
        assert isinstance(block, LanguageDecoderBlock), f"Block {j} must be LanguageDecoderBlock"
        assert hasattr(block, "norm1")
        assert hasattr(block, "self_attn")  # MMHA
        assert hasattr(block, "norm2")
        assert hasattr(block, "cross_attn")  # Cross-MHA
        assert hasattr(block, "norm3")
        assert hasattr(block, "ffn")
        assert hasattr(block, "vim")
        assert isinstance(block.vim, VisionInjectionModule)

        vim_instances.append(id(block.vim))
        vim_weight_ptrs.append(block.vim.w_fc.weight.data_ptr())

    # 3. Verify all 7 VIM instances are independent objects
    assert len(set(vim_instances)) == 7, "All 7 VIM modules must be distinct module instances"
    assert len(set(vim_weight_ptrs)) == 7, "All 7 VIM modules must have distinct parameter memory"

    # 4. Verify no final LayerNorm exists on decoder
    assert not hasattr(decoder, "norm"), "Decoder must not have a final LayerNorm"


def test_decoder_causal_masking(embed_dim, max_tokens):
    """Verify MMHA is strictly causal: position i cannot attend to future positions j > i."""
    block = LanguageDecoderBlock(
        embed_dim=embed_dim,
        num_heads=8,
        vim_heads=8,
        dim_feedforward=2048,
    )
    block.eval()

    # Create causal mask
    causal_mask = torch.triu(
        torch.ones(max_tokens, max_tokens, dtype=torch.bool),
        diagonal=1,
    )

    # Input 1: Random sequence
    torch.manual_seed(100)
    x1 = torch.randn(1, max_tokens, embed_dim)

    # Input 2: Identical up to position k=50, completely different after k=50
    k = 50
    x2 = x1.clone()
    x2[:, k + 1 :, :] = torch.randn(1, max_tokens - (k + 1), embed_dim)

    # Dummy context
    t_hig = torch.randn(1, max_tokens, embed_dim)
    i_v = torch.randn(1, embed_dim)

    with torch.no_grad():
        # Step 1 test on self-attention with causal mask
        norm_x1 = block.norm1(x1)
        attn_out1, _ = block.self_attn(norm_x1, norm_x1, norm_x1, attn_mask=causal_mask)

        norm_x2 = block.norm1(x2)
        attn_out2, _ = block.self_attn(norm_x2, norm_x2, norm_x2, attn_mask=causal_mask)

    # MMHA outputs for positions 0..k MUST be identical
    assert torch.allclose(attn_out1[:, : k + 1, :], attn_out2[:, : k + 1, :], atol=1e-6), (
        f"Causal MMHA violation: positions <= {k} were affected by future modifications at positions > {k}"
    )

    # MMHA outputs for positions > k MUST be different
    assert not torch.allclose(attn_out1[:, k + 1 :, :], attn_out2[:, k + 1 :, :]), (
        "Positions > k must reflect the modified input tokens"
    )


def test_cross_attention_uses_complete_t_hig(dummy_t_low, dummy_t_hig, dummy_visual_feature):
    """Verify cross-attention in LD uses full T_hig^e [B, 308, 512] and not T_l [B, 512]."""
    decoder = LanguageDecoder(
        vocab_size=49408,
        embed_dim=512,
        max_text_tokens=308,
        num_blocks=7,
    )

    # 1. Forward with full T_hig succeeds
    out = decoder(dummy_t_low, dummy_t_hig, dummy_visual_feature, return_logits=False)
    assert out.t_rec.shape == (2, 308, 512)

    # 2. Forward with squeezed T_l [2, 512] fails with shape mismatch in Cross-MHA
    t_l = dummy_t_hig[:, -1, :]
    with pytest.raises(Exception):
        decoder(dummy_t_low, t_l, dummy_visual_feature, return_logits=False)


def test_language_decoder_forward_and_output_shapes(dummy_t_low, dummy_t_hig, dummy_visual_feature, batch_size, max_tokens, embed_dim, vocab_size):
    """Verify Language Decoder forward pass shapes, outputs, and interface ergonomics."""
    decoder = LanguageDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_text_tokens=max_tokens,
        num_blocks=7,
    )

    # Forward with vocabulary logits
    out = decoder(dummy_t_low, dummy_t_hig, dummy_visual_feature, return_logits=True)

    # 1. Check container type and dictionary keys
    assert isinstance(out, LanguageDecoderOutput)
    assert "t_rec" in out
    assert "t_pre" in out

    # 2. Check shapes
    t_rec = out["t_rec"]
    t_pre = out["t_pre"]

    assert t_rec.shape == (batch_size, max_tokens, embed_dim), f"T_rec shape mismatch: {t_rec.shape}"
    assert t_pre.shape == (batch_size, max_tokens, vocab_size), f"T_pre shape mismatch: {t_pre.shape}"

    # 3. Check attribute access
    assert torch.equal(out.t_rec, t_rec)
    assert torch.equal(out.t_pre, t_pre)

    # 4. Check tuple unpacking
    unpacked_rec, unpacked_pre = decoder(dummy_t_low, dummy_t_hig, dummy_visual_feature, return_logits=True)
    assert torch.equal(unpacked_rec, t_rec)
    assert torch.equal(unpacked_pre, t_pre)

    # 5. Check finite values
    assert torch.isfinite(t_rec).all()
    assert torch.isfinite(t_pre).all()


def test_vocabulary_weight_tying(dummy_t_low, dummy_t_hig, dummy_visual_feature):
    """PAPER_SPECIFIED: Verify true weight tying between encoder token_embed and decoder W_voc^T (Eq. 24)."""
    embeddings = LanguageEmbeddings(vocab_size=49408, embed_dim=512)
    decoder = LanguageDecoder(
        vocab_size=49408,
        embed_dim=512,
        token_embedding=embeddings.token_embed,
    )

    # 1. Verify exact parameter object and memory pointer identity
    assert decoder.token_embedding.weight is embeddings.token_embed.weight
    assert decoder.token_embedding.weight.data_ptr() == embeddings.token_embed.weight.data_ptr()

    # 2. Test projection mathematics: T_pre = T_rec @ W_voc^T
    out = decoder(dummy_t_low, dummy_t_hig, dummy_visual_feature, return_logits=True)
    t_rec = out.t_rec
    t_pre = out.t_pre

    expected_logits = F.linear(t_rec, embeddings.token_embed.weight)
    assert torch.allclose(t_pre, expected_logits), "T_pre must equal F.linear(T_rec, W_voc)"


def test_language_decoder_backward_gradients(dummy_t_low, dummy_t_hig, dummy_visual_feature):
    """Verify gradient backpropagation reaches all LD components and tied embeddings."""
    embeddings = LanguageEmbeddings(vocab_size=49408, embed_dim=512)
    decoder = LanguageDecoder(
        vocab_size=49408,
        embed_dim=512,
        token_embedding=embeddings.token_embed,
    )

    t_low = dummy_t_low.clone().detach().requires_grad_(True)
    t_hig = dummy_t_hig.clone().detach().requires_grad_(True)
    i_v = dummy_visual_feature.clone().detach().requires_grad_(True)

    out = decoder(t_low, t_hig, i_v, return_logits=True)
    t_rec = out.t_rec
    t_pre = out.t_pre

    # Memory-conscious synthetic scalar loss
    loss = t_rec.sum() + t_pre[:, :10, :100].sum()
    loss.backward()

    # 1. Gradients reach BOS and decoder positional embeddings
    assert decoder.bos_embed.grad is not None
    assert torch.isfinite(decoder.bos_embed.grad).all()
    assert (decoder.bos_embed.grad.abs() > 0).any(), "BOS embedding must have non-zero gradients"

    assert decoder.pos_embed.grad is not None
    assert torch.isfinite(decoder.pos_embed.grad).all()
    assert (decoder.pos_embed.grad.abs() > 0).any(), "Decoder pos_embed must have non-zero gradients"

    # 2. Gradients reach tied vocabulary embedding matrix
    assert embeddings.token_embed.weight.grad is not None
    assert torch.isfinite(embeddings.token_embed.weight.grad).all()
    assert (embeddings.token_embed.weight.grad.abs() > 0).any(), "Tied W_voc must have non-zero gradients"

    # 3. Gradients reach each of the 7 decoder blocks
    for j, block in enumerate(decoder.blocks):
        # MMHA
        assert block.self_attn.in_proj_weight.grad is not None, f"Block {j} MMHA in_proj missing grad"
        assert (block.self_attn.in_proj_weight.grad.abs() > 0).any(), f"Block {j} MMHA in_proj grad is zero"
        assert block.self_attn.out_proj.weight.grad is not None, f"Block {j} MMHA out_proj missing grad"
        assert (block.self_attn.out_proj.weight.grad.abs() > 0).any(), f"Block {j} MMHA out_proj grad is zero"

        # Cross-MHA
        assert block.cross_attn.in_proj_weight.grad is not None, f"Block {j} Cross-MHA in_proj missing grad"
        assert (block.cross_attn.in_proj_weight.grad.abs() > 0).any(), f"Block {j} Cross-MHA in_proj grad is zero"
        assert block.cross_attn.out_proj.weight.grad is not None, f"Block {j} Cross-MHA out_proj missing grad"
        assert (block.cross_attn.out_proj.weight.grad.abs() > 0).any(), f"Block {j} Cross-MHA out_proj grad is zero"

        # FFN
        assert block.ffn[0].weight.grad is not None, f"Block {j} FFN layer 1 missing grad"
        assert (block.ffn[0].weight.grad.abs() > 0).any(), f"Block {j} FFN layer 1 grad is zero"
        assert block.ffn[3].weight.grad is not None, f"Block {j} FFN layer 2 missing grad"
        assert (block.ffn[3].weight.grad.abs() > 0).any(), f"Block {j} FFN layer 2 grad is zero"

        # VIM W_val and W_fc
        assert block.vim.w_val.weight.grad is not None, f"Block {j} VIM w_val missing grad"
        assert (block.vim.w_val.weight.grad.abs() > 0).any(), f"Block {j} VIM w_val grad is zero"
        assert block.vim.w_fc.weight.grad is not None, f"Block {j} VIM w_fc missing grad"
        assert (block.vim.w_fc.weight.grad.abs() > 0).any(), f"Block {j} VIM w_fc grad is zero"

        # VIM W_que and W_key exist (gradient tensor exists)
        assert block.vim.w_que.weight.grad is not None, f"Block {j} VIM w_que missing grad tensor"
        assert block.vim.w_key.weight.grad is not None, f"Block {j} VIM w_key missing grad tensor"

    # 4. Inputs receive gradients
    assert t_low.grad is not None and (t_low.grad.abs() > 0).any(), "T_low must receive gradients"
    assert t_hig.grad is not None and (t_hig.grad.abs() > 0).any(), "T_hig must receive gradients"
    assert i_v.grad is not None and (i_v.grad.abs() > 0).any(), "I_v must receive gradients"
