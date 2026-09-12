"""Tests for Language Encoder (LE), LanguageEmbeddings, and VIM integration (Phase 6).

PAPER_SPECIFIED:
- Vocabulary size: s = 49,408
- Maximum token length: n = 308
- Embedding dimension: d = 512
- Language Encoder depth: E = 12 blocks (Section III-D, Eq. 3)
- Sequential block flow: MHA -> VIM -> FFN
- Contextualized representation: T_hig^e in R^(B x 308 x 512)
- Global language representation: T_l in R^(B x 512) is strictly the LAST TOKEN of T_hig^e (T_hig[:, -1, :])
"""

import pytest
import torch
import torch.nn as nn

from models.language.embeddings import LanguageEmbeddings
from models.language.vim import VisionInjectionModule, VIM
from models.language.language_encoder import (
    LanguageEncoderBlock,
    LanguageEncoder,
    LanguageEncoderOutput,
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
def dummy_token_ids(batch_size, max_tokens, vocab_size):
    """Generate synthetic token IDs in [0, vocab_size - 1]."""
    torch.manual_seed(42)
    return torch.randint(0, vocab_size, (batch_size, max_tokens), dtype=torch.long)


@pytest.fixture
def dummy_visual_feature(batch_size, embed_dim):
    """Generate synthetic visual feature I_v in R^(B x 512)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, embed_dim)


def test_language_embeddings(dummy_token_ids, batch_size, max_tokens, embed_dim, vocab_size):
    """Test token embedding interface and positional embeddings."""
    embed_layer = LanguageEmbeddings(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_text_tokens=max_tokens,
    )

    t_low, t_1_tra = embed_layer(dummy_token_ids)

    # Verify shapes
    assert t_low.shape == (batch_size, max_tokens, embed_dim), (
        f"Expected t_low shape {(batch_size, max_tokens, embed_dim)}, got {t_low.shape}"
    )
    assert t_1_tra.shape == (batch_size, max_tokens, embed_dim), (
        f"Expected t_1_tra shape {(batch_size, max_tokens, embed_dim)}, got {t_1_tra.shape}"
    )
    assert embed_layer.pos_embed.shape == (1, max_tokens, embed_dim), (
        f"Expected pos_embed shape {(1, max_tokens, embed_dim)}, got {embed_layer.pos_embed.shape}"
    )

    # Verify T_1^tra = T_low^e + P_e
    expected_t_1 = t_low + embed_layer.pos_embed
    assert torch.allclose(t_1_tra, expected_t_1), "t_1_tra must equal t_low + pos_embed"

    # Verify finiteness
    assert torch.isfinite(t_low).all()
    assert torch.isfinite(t_1_tra).all()


def test_language_encoder_structure():
    """Verify Language Encoder has exactly E=12 blocks and independent VIM instances."""
    encoder = LanguageEncoder(
        vocab_size=49408,
        embed_dim=512,
        max_text_tokens=308,
        num_blocks=12,
        num_heads=8,
        vim_heads=8,
        dim_feedforward=2048,
    )

    # Verify exactly 12 blocks
    assert len(encoder.blocks) == 12, f"Expected 12 blocks, got {len(encoder.blocks)}"
    assert encoder.num_blocks == 12

    # Verify each block contains MHA, VIM, and FFN
    vim_instances = []
    vim_weight_ptrs = []
    for j, block in enumerate(encoder.blocks):
        assert isinstance(block, LanguageEncoderBlock), f"Block {j} must be LanguageEncoderBlock"
        assert hasattr(block, "norm1")
        assert hasattr(block, "self_attn")
        assert hasattr(block, "vim")
        assert isinstance(block.vim, VisionInjectionModule)
        assert hasattr(block, "norm2")
        assert hasattr(block, "ffn")

        vim_instances.append(id(block.vim))
        vim_weight_ptrs.append(block.vim.w_fc.weight.data_ptr())

    # Verify all 12 VIM instances are distinct module objects (no cross-layer sharing)
    assert len(set(vim_instances)) == 12, "All 12 VIM modules must be distinct module instances"
    assert len(set(vim_weight_ptrs)) == 12, "All 12 VIM modules must have distinct parameter memory"


def test_language_encoder_block_flow(batch_size, max_tokens, embed_dim, dummy_visual_feature):
    """Verify forward pass of a single LanguageEncoderBlock preserves tensor shape."""
    block = LanguageEncoderBlock(
        embed_dim=embed_dim,
        num_heads=8,
        vim_heads=8,
        dim_feedforward=2048,
    )

    x = torch.randn(batch_size, max_tokens, embed_dim)
    out = block(x, dummy_visual_feature)

    assert out.shape == (batch_size, max_tokens, embed_dim)
    assert torch.isfinite(out).all()

    # Also test with [B, 1, 512] visual feature
    i_v_3d = dummy_visual_feature.unsqueeze(1)
    out_3d = block(x, i_v_3d)
    assert out_3d.shape == (batch_size, max_tokens, embed_dim)
    assert torch.allclose(out, out_3d, atol=1e-6)


def test_language_encoder_forward(dummy_token_ids, dummy_visual_feature, batch_size, max_tokens, embed_dim):
    """Verify Language Encoder full forward pass shapes, outputs, and interface ergonomics."""
    encoder = LanguageEncoder(
        vocab_size=49408,
        embed_dim=embed_dim,
        max_text_tokens=max_tokens,
        num_blocks=12,
        num_heads=8,
        vim_heads=8,
        dim_feedforward=2048,
    )

    out = encoder(dummy_token_ids, dummy_visual_feature)

    # 1. Check container type and dictionary keys
    assert isinstance(out, LanguageEncoderOutput)
    assert "t_low" in out
    assert "t_hig" in out
    assert "t_l" in out

    # 2. Check shapes
    t_low = out["t_low"]
    t_hig = out["t_hig"]
    t_l = out["t_l"]

    assert t_low.shape == (batch_size, max_tokens, embed_dim), f"T_low shape mismatch: {t_low.shape}"
    assert t_hig.shape == (batch_size, max_tokens, embed_dim), f"T_hig shape mismatch: {t_hig.shape}"
    assert t_l.shape == (batch_size, embed_dim), f"T_l shape mismatch: {t_l.shape}"

    # 3. Check attribute access
    assert torch.equal(out.t_low, t_low)
    assert torch.equal(out.t_hig, t_hig)
    assert torch.equal(out.t_l, t_l)

    # 4. Check tuple unpacking
    unpacked_low, unpacked_hig, unpacked_l = encoder(dummy_token_ids, dummy_visual_feature)
    assert torch.equal(unpacked_low, t_low)
    assert torch.equal(unpacked_hig, t_hig)
    assert torch.equal(unpacked_l, t_l)

    # 5. Check finite values
    assert torch.isfinite(t_low).all()
    assert torch.isfinite(t_hig).all()
    assert torch.isfinite(t_l).all()


def test_last_token_global_language_representation(dummy_token_ids, dummy_visual_feature):
    """PAPER_SPECIFIED test: Verify T_l is strictly the LAST TOKEN of T_hig^e (T_hig[:, -1, :])."""
    encoder = LanguageEncoder(
        vocab_size=49408,
        embed_dim=512,
        max_text_tokens=308,
        num_blocks=12,
    )

    out = encoder(dummy_token_ids, dummy_visual_feature)
    t_hig = out.t_hig
    t_l = out.t_l

    # 1. Exact equality with last token
    expected_t_l = t_hig[:, -1, :]
    assert torch.equal(t_l, expected_t_l), "T_l MUST be identically equal to T_hig[:, -1, :]"

    # 2. Verify T_l is NOT mean pooling
    mean_pooled = t_hig.mean(dim=1)
    assert not torch.allclose(t_l, mean_pooled), "T_l must NOT be computed via mean pooling"

    # 3. Verify T_l is NOT max pooling
    max_pooled = t_hig.max(dim=1).values
    assert not torch.allclose(t_l, max_pooled), "T_l must NOT be computed via max pooling"

    # 4. Verify T_l is NOT the first token
    first_token = t_hig[:, 0, :]
    assert not torch.allclose(t_l, first_token), "T_l must NOT be the first token"


def test_vim_singleton_visual_kv_behavior(dummy_token_ids, dummy_visual_feature):
    """Verify VIM modules receive I_v and retain singleton Key/Value mathematical properties."""
    encoder = LanguageEncoder(
        vocab_size=49408,
        embed_dim=512,
        max_text_tokens=308,
        num_blocks=12,
    )

    # Check VIM modules in all 12 blocks
    for j, block in enumerate(encoder.blocks):
        # Input test to block's VIM
        t_tok = torch.randn(2, 308, 512)
        t_add, attn = block.vim(t_tok, dummy_visual_feature, return_attention=True)

        assert t_add.shape == (2, 308, 512)
        # Singleton softmax probability must be exactly 1.0 across all heads and tokens
        assert attn.shape == (2, 8, 308, 1)
        assert torch.allclose(attn, torch.ones_like(attn)), (
            f"Block {j} VIM attention probs must be 1.0 for singleton visual feature"
        )


def test_language_encoder_backward_gradients(dummy_token_ids, dummy_visual_feature):
    """Verify gradient backpropagation reaches all LE components."""
    encoder = LanguageEncoder(
        vocab_size=49408,
        embed_dim=512,
        max_text_tokens=308,
        num_blocks=12,
    )

    # Enable gradients on visual input to check end-to-end flow
    i_v = dummy_visual_feature.clone().detach().requires_grad_(True)

    out = encoder(dummy_token_ids, i_v)
    t_low = out.t_low
    t_hig = out.t_hig
    t_l = out.t_l

    # Synthetic scalar loss involving both T_l and T_hig
    loss = t_l.sum() + t_hig.sum()
    loss.backward()

    # 1. Gradients reach token embeddings
    assert encoder.embeddings.token_embed.weight.grad is not None
    assert torch.isfinite(encoder.embeddings.token_embed.weight.grad).all()

    # 2. Gradients reach positional embeddings
    assert encoder.embeddings.pos_embed.grad is not None
    assert torch.isfinite(encoder.embeddings.pos_embed.grad).all()
    assert (encoder.embeddings.pos_embed.grad.abs() > 0).any(), "Positional embedding must have non-zero gradients"

    # 3. Gradients reach each of the 12 blocks
    for j, block in enumerate(encoder.blocks):
        # Self-Attention
        assert block.self_attn.in_proj_weight.grad is not None, f"Block {j} self_attn in_proj missing grad"
        assert (block.self_attn.in_proj_weight.grad.abs() > 0).any(), f"Block {j} self_attn in_proj grad is all zero"
        assert block.self_attn.out_proj.weight.grad is not None, f"Block {j} self_attn out_proj missing grad"
        assert (block.self_attn.out_proj.weight.grad.abs() > 0).any(), f"Block {j} self_attn out_proj grad is all zero"

        # FFN
        assert block.ffn[0].weight.grad is not None, f"Block {j} FFN layer 1 missing grad"
        assert (block.ffn[0].weight.grad.abs() > 0).any(), f"Block {j} FFN layer 1 grad is all zero"
        assert block.ffn[3].weight.grad is not None, f"Block {j} FFN layer 2 missing grad"
        assert (block.ffn[3].weight.grad.abs() > 0).any(), f"Block {j} FFN layer 2 grad is all zero"

        # VIM W_val and W_fc (Value projection and output projection)
        assert block.vim.w_val.weight.grad is not None, f"Block {j} VIM w_val missing grad"
        assert (block.vim.w_val.weight.grad.abs() > 0).any(), f"Block {j} VIM w_val grad is all zero"
        assert block.vim.w_fc.weight.grad is not None, f"Block {j} VIM w_fc missing grad"
        assert (block.vim.w_fc.weight.grad.abs() > 0).any(), f"Block {j} VIM w_fc grad is all zero"

        # Note: As established in Phase 5, W_que and W_key receive zero gradients
        # because the softmax derivative of a singleton key is identically zero.
        assert block.vim.w_que.weight.grad is not None, f"Block {j} VIM w_que missing grad tensor"
        assert block.vim.w_key.weight.grad is not None, f"Block {j} VIM w_key missing grad tensor"

    # 5. Visual input receives gradient back from VIMs
    assert i_v.grad is not None
    assert (i_v.grad.abs() > 0).any(), "Visual feature I_v must receive non-zero gradients from LE blocks"
