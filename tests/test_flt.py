"""Tests for Fine-grained Language Transformer (FLT) wrapper (Phase 8).

PAPER_SPECIFIED:
- Integrates Language Encoder (E=12) and Language Decoder (D=7).
- True weight tying: Decoder vocabulary projection uses transposed encoder vocabulary embedding W_voc^T.
- Produces T_low, T_hig, T_l, T_rec, and T_pre.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.language.flt import FineGrainedLanguageTransformer, FLT, FLTOutput


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
    """Synthetic token IDs in [0, vocab_size - 1]."""
    torch.manual_seed(42)
    return torch.randint(0, vocab_size, (batch_size, max_tokens), dtype=torch.long)


@pytest.fixture
def dummy_visual_feature(batch_size, embed_dim):
    """Synthetic visual feature I_v in R^(B x 512)."""
    torch.manual_seed(43)
    return torch.randn(batch_size, embed_dim)


def test_flt_initialization_and_weight_tying(vocab_size, embed_dim, max_tokens):
    """Verify FLT architecture, submodule depths (E=12, D=7), and true parameter sharing."""
    flt = FLT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_text_tokens=max_tokens,
        encoder_blocks=12,
        decoder_blocks=7,
    )

    # 1. Block depths
    assert len(flt.encoder.blocks) == 12
    assert len(flt.decoder.blocks) == 7

    # 2. True parameter sharing: decoder token_embedding is identical to encoder embeddings.token_embed
    assert flt.decoder.token_embedding.weight is flt.encoder.embeddings.token_embed.weight
    assert flt.decoder.token_embedding.weight.data_ptr() == flt.encoder.embeddings.token_embed.weight.data_ptr()


def test_flt_forward_shapes_and_outputs(dummy_token_ids, dummy_visual_feature, batch_size, max_tokens, embed_dim, vocab_size):
    """Verify FLT end-to-end forward pass shapes, outputs, and interface container."""
    flt = FLT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_text_tokens=max_tokens,
    )

    out = flt(dummy_token_ids, dummy_visual_feature, return_logits=True)

    # 1. Container type
    assert isinstance(out, FLTOutput)
    assert "t_low" in out
    assert "t_hig" in out
    assert "t_l" in out
    assert "t_rec" in out
    assert "t_pre" in out

    # 2. Shapes
    assert out.t_low.shape == (batch_size, max_tokens, embed_dim)
    assert out.t_hig.shape == (batch_size, max_tokens, embed_dim)
    assert out.t_l.shape == (batch_size, embed_dim)
    assert out.t_rec.shape == (batch_size, max_tokens, embed_dim)
    assert out.t_pre.shape == (batch_size, max_tokens, vocab_size)

    # 3. Tuple unpacking
    t_low, t_hig, t_l, t_rec, t_pre = flt(dummy_token_ids, dummy_visual_feature, return_logits=True)
    assert torch.equal(t_low, out.t_low)
    assert torch.equal(t_hig, out.t_hig)
    assert torch.equal(t_l, out.t_l)
    assert torch.equal(t_rec, out.t_rec)
    assert torch.equal(t_pre, out.t_pre)

    # 4. Last-token identity
    assert torch.equal(out.t_l, out.t_hig[:, -1, :])

    # 5. Finiteness
    assert torch.isfinite(out.t_low).all()
    assert torch.isfinite(out.t_hig).all()
    assert torch.isfinite(out.t_l).all()
    assert torch.isfinite(out.t_rec).all()
    assert torch.isfinite(out.t_pre).all()


def test_flt_encode_and_decode_methods(dummy_token_ids, dummy_visual_feature):
    """Verify separate encode() and decode() methods produce identical results to forward()."""
    flt = FLT(vocab_size=49408, embed_dim=512, max_text_tokens=308)
    flt.eval()

    with torch.no_grad():
        # Step-by-step
        enc_out = flt.encode(dummy_token_ids, dummy_visual_feature)
        dec_out = flt.decode(enc_out.t_low, enc_out.t_hig, dummy_visual_feature, return_logits=True)

        # Monolithic forward
        fwd_out = flt(dummy_token_ids, dummy_visual_feature, return_logits=True)

    assert torch.allclose(enc_out.t_low, fwd_out.t_low)
    assert torch.allclose(enc_out.t_hig, fwd_out.t_hig)
    assert torch.allclose(enc_out.t_l, fwd_out.t_l)
    assert torch.allclose(dec_out.t_rec, fwd_out.t_rec)
    assert torch.allclose(dec_out.t_pre, fwd_out.t_pre)


def test_flt_backward_gradients(dummy_token_ids, dummy_visual_feature):
    """Verify gradients backpropagate through entire FLT pipeline."""
    flt = FLT(vocab_size=49408, embed_dim=512, max_text_tokens=308)

    i_v = dummy_visual_feature.clone().detach().requires_grad_(True)
    out = flt(dummy_token_ids, i_v, return_logits=True)

    # Memory-conscious loss
    loss = out.t_l.sum() + out.t_rec.sum() + out.t_pre[:, :5, :50].sum()
    loss.backward()

    # 1. Embeddings & tied vocabulary
    assert flt.token_embedding.weight.grad is not None
    assert (flt.token_embedding.weight.grad.abs() > 0).any()

    # 2. Positional embeddings (Encoder & Decoder)
    assert flt.encoder.embeddings.pos_embed.grad is not None
    assert flt.decoder.pos_embed.grad is not None
    assert flt.decoder.bos_embed.grad is not None

    # 3. Vision input receives gradient
    assert i_v.grad is not None
    assert (i_v.grad.abs() > 0).any()
