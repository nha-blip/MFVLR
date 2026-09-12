"""Tests for MFVLR paper-faithful image-only inference mode (Phase 10).

PAPER_SPECIFIED:
- Standard inference is image-only: I -> MVE + VD -> I_v -> Detection Head -> y_pre, m_pre.
- No prompt, no token_ids, no tokenizer, and no language path (LE, LD, FLT, Adapter, CMC, KL) are executed.
- Predictions:
    Detection: argmax(y_pre, dim=1) in {0, 1}^B
    Localization: argmax(m_pre, dim=1) in {0, 1}^(B x 224 x 224)
"""

import pytest
import torch
import torch.nn as nn

from models.mfvlr import MFVLR, MFVLRInferenceOutput


@pytest.fixture
def dummy_image():
    """Synthetic image input [2, 3, 224, 224]."""
    torch.manual_seed(42)
    return torch.randn(2, 3, 224, 224).clamp(0.0, 1.0)


@pytest.fixture
def dummy_token_ids():
    """Synthetic prompt token IDs [2, 308]."""
    torch.manual_seed(42)
    return torch.randint(0, 49408, (2, 308), dtype=torch.long)


def test_image_only_forward_shapes_and_predictions(dummy_image):
    """PAPER_SPECIFIED test: Verify image-only inference forward pass shapes and predictions."""
    model = MFVLR()
    model.eval()

    # Image-only forward without prompt / token_ids
    out: MFVLRInferenceOutput = model.forward_image_only(dummy_image, return_intermediates=True)

    # 1. Output tensor shapes
    assert out.y_pre.shape == (2, 2)
    assert out.m_pre.shape == (2, 2, 224, 224)
    assert out.i_pre.shape == (2, 3, 224, 224)
    assert out.i_r.shape == (2, 3, 224, 224)
    assert out.i_v.shape == (2, 512)
    assert out.i_g.shape == (2, 512)
    assert out.i_rg.shape == (2, 512)
    assert out.i_loc.shape == (2, 1024, 14, 14)

    # 2. Prediction helpers
    pred_class = out.predict_class()
    assert pred_class.shape == (2,)
    assert pred_class.dtype == torch.long
    assert ((pred_class == 0) | (pred_class == 1)).all()

    fake_prob = out.predict_fake_prob()
    assert fake_prob.shape == (2,)
    assert (fake_prob >= 0.0).all() and (fake_prob <= 1.0).all()

    pred_mask = out.predict_mask()
    assert pred_mask.shape == (2, 224, 224)
    assert pred_mask.dtype == torch.long
    assert ((pred_mask == 0) | (pred_mask == 1)).all()


def test_image_only_strictly_does_not_execute_flt(dummy_image, monkeypatch):
    """CRITICAL PAPER_SPECIFIED test: Prove FLT is never invoked during image-only inference."""
    model = MFVLR()
    model.eval()

    def raise_flt_error(*args, **kwargs):
        raise RuntimeError("FLT must NOT be executed during image-only inference!")

    # Replace FLT forward with an error-raising function
    monkeypatch.setattr(model.flt, "forward", raise_flt_error)

    # Calling image-only inference MUST succeed without triggering FLT
    out = model.forward_image_only(dummy_image)
    assert out.y_pre is not None
    assert out.m_pre is not None


def test_eval_mode_vision_equivalence_between_full_and_image_only(dummy_image, dummy_token_ids):
    """PAPER_SPECIFIED test: Verify full forward vision outputs and image-only forward vision outputs are identical."""
    model = MFVLR()
    model.eval()

    with torch.no_grad():
        # Full training forward pass with token IDs
        full_out = model(dummy_image, dummy_token_ids, return_logits=False)

        # Image-only inference forward pass
        img_out = model.forward_image_only(dummy_image, return_intermediates=True)

    # All vision and detection outputs must be mathematically identical
    assert torch.allclose(full_out.y_pre, img_out.y_pre, atol=1e-6)
    assert torch.allclose(full_out.m_pre, img_out.m_pre, atol=1e-6)
    assert torch.allclose(full_out.i_pre, img_out.i_pre, atol=1e-6)
    assert torch.allclose(full_out.i_r, img_out.i_r, atol=1e-6)
    assert torch.allclose(full_out.i_v, img_out.i_v, atol=1e-6)
    assert torch.allclose(full_out.i_g, img_out.i_g, atol=1e-6)
    assert torch.allclose(full_out.i_rg, img_out.i_rg, atol=1e-6)
    assert torch.allclose(full_out.i_loc, img_out.i_loc, atol=1e-6)


def test_residual_and_fusion_exactness_in_image_only(dummy_image):
    """Verify I_r = |I_pre - I| and I_v = I_g + I_rg in image-only inference."""
    model = MFVLR()
    model.eval()

    with torch.no_grad():
        out = model.forward_image_only(dummy_image, return_intermediates=True)

    expected_ir = torch.abs(out.i_pre - dummy_image)
    assert torch.allclose(out.i_r, expected_ir, atol=1e-7)

    expected_iv = out.i_g + out.i_rg
    assert torch.allclose(out.i_v, expected_iv, atol=1e-7)
