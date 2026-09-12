"""Unit tests for Vision Decoder (VD), Appearance Decoder (AD), Mask Decoder (MD), and residual generation.

PAPER_SPECIFIED constraints tested:
1. Input I_loc: [2, 1024, 14, 14]
2. Reconstructed appearance I_pre: [2, 3, 224, 224]
3. Predicted mask logits M_pre: [2, 2, 224, 224]
4. Appearance values finite and in range [0, 1] (Sigmoid assumption)
5. Mask output finite raw logits (no internal softmax)
6. Exact SAME decoder trunk module used by AD and MD
7. True parameter sharing (identical parameter objects across AD/MD decoder trunk)
8. Separate convolution heads for AD (3 channels) and MD (2 channels)
9. Residual image: I_r = |I_pre - I|, shape [2, 3, 224, 224], non-negative (I_r >= 0)
10. Gradient backpropagation through joint appearance and mask loss reaches shared trunk and both heads
"""

import pytest
import torch
import torch.nn as nn

from models.vision.unet_decoder import UNetDecoderTrunk, UpsampleBlock
from models.vision.appearance_decoder import AppearanceDecoder, AppearanceHead
from models.vision.mask_decoder import MaskDecoder, MaskHead
from models.vision.vision_decoder import VisionDecoder


def test_unet_decoder_trunk_output_shape():
    """Verify UNetDecoderTrunk decodes [B, 1024, 14, 14] to [B, 64, 224, 224]."""
    trunk = UNetDecoderTrunk(in_channels=1024, channels=(512, 256, 128, 64))
    i_loc = torch.randn(2, 1024, 14, 14)

    # Without skips
    out_no_skips = trunk(i_loc, skips=None)
    assert out_no_skips.shape == (2, 64, 224, 224)
    assert torch.isfinite(out_no_skips).all()

    # With multi-scale skips
    s0 = torch.randn(2, 64, 224, 224)
    s1 = torch.randn(2, 128, 112, 112)
    s2 = torch.randn(2, 256, 56, 56)
    s3 = torch.randn(2, 512, 28, 28)
    out_with_skips = trunk(i_loc, skips=[s0, s1, s2, s3])
    assert out_with_skips.shape == (2, 64, 224, 224)
    assert torch.isfinite(out_with_skips).all()


def test_appearance_decoder_output():
    """Verify Appearance Decoder produces I_pre [2, 3, 224, 224] in [0, 1]."""
    trunk = UNetDecoderTrunk(in_channels=1024, channels=(512, 256, 128, 64))
    ad = AppearanceDecoder(decoder_trunk=trunk, out_channels=3)

    i_loc = torch.randn(2, 1024, 14, 14)
    i_pre = ad(i_loc)

    assert i_pre.shape == (2, 3, 224, 224)
    assert torch.isfinite(i_pre).all()
    # Sigmoid assumption constraint
    assert (i_pre >= 0.0).all()
    assert (i_pre <= 1.0).all()


def test_mask_decoder_output():
    """Verify Mask Decoder produces raw logits M_pre [2, 2, 224, 224]."""
    trunk = UNetDecoderTrunk(in_channels=1024, channels=(512, 256, 128, 64))
    md = MaskDecoder(decoder_trunk=trunk, num_classes=2)

    i_loc = torch.randn(2, 1024, 14, 14)
    m_pre = md(i_loc)

    assert m_pre.shape == (2, 2, 224, 224)
    assert torch.isfinite(m_pre).all()


def test_vision_decoder_true_weight_sharing():
    """Verify AD and MD share the exact SAME U-Net decoder trunk instance and parameters."""
    vd = VisionDecoder(
        in_channels=1024,
        out_image_channels=3,
        num_mask_classes=2,
    )

    # 1. Exact object identity
    assert vd.ad.decoder_trunk is vd.md.decoder_trunk
    assert vd.ad.decoder_trunk is vd.decoder_trunk
    assert vd.md.decoder_trunk is vd.decoder_trunk

    # 2. Parameter identity across decoder trunk
    ad_trunk_params = list(vd.ad.decoder_trunk.parameters())
    md_trunk_params = list(vd.md.decoder_trunk.parameters())
    assert len(ad_trunk_params) == len(md_trunk_params)
    for p_ad, p_md in zip(ad_trunk_params, md_trunk_params):
        assert p_ad is p_md

    # 3. Heads must be separate objects
    assert vd.ad.head is not vd.md.head
    assert vd.ad.head.conv.weight is not vd.md.head.conv.weight
    assert vd.ad.head.conv.out_channels == 3
    assert vd.md.head.conv.out_channels == 2


def test_vision_decoder_residual_generation():
    """Verify residual generation I_r = |I_pre - I| is non-negative and shape [2, 3, 224, 224]."""
    vd = VisionDecoder(
        in_channels=1024,
        out_image_channels=3,
        num_mask_classes=2,
    )

    i_loc = torch.randn(2, 1024, 14, 14)
    orig_image = torch.rand(2, 3, 224, 224)

    outputs = vd(i_loc, orig_image=orig_image)

    assert "i_pre" in outputs
    assert "m_pre" in outputs
    assert "i_r" in outputs

    assert outputs["i_pre"].shape == (2, 3, 224, 224)
    assert outputs["m_pre"].shape == (2, 2, 224, 224)
    assert outputs["i_r"].shape == (2, 3, 224, 224)

    # Verify residual correctness
    expected_ir = torch.abs(outputs["i_pre"] - orig_image)
    assert torch.allclose(outputs["i_r"], expected_ir, atol=1e-6)
    assert (outputs["i_r"] >= 0.0).all()
    assert torch.isfinite(outputs["i_r"]).all()


def test_vision_decoder_backpropagation():
    """Verify gradients propagate to shared trunk, appearance head, and mask head."""
    vd = VisionDecoder(
        in_channels=1024,
        out_image_channels=3,
        num_mask_classes=2,
    )

    i_loc = torch.randn(2, 1024, 14, 14, requires_grad=True)
    orig_image = torch.rand(2, 3, 224, 224, requires_grad=True)

    outputs = vd(i_loc, orig_image=orig_image)
    i_pre = outputs["i_pre"]
    m_pre = outputs["m_pre"]
    i_r = outputs["i_r"]

    # Scalar loss combining appearance reconstruction and mask localization
    loss = i_pre.sum() + m_pre.sum() + i_r.sum()
    loss.backward()

    # 1. Check shared trunk gradients
    assert vd.decoder_trunk.up1.conv.conv1.weight.grad is not None
    assert torch.isfinite(vd.decoder_trunk.up1.conv.conv1.weight.grad).all()

    # 2. Check appearance head gradients
    assert vd.ad.head.conv.weight.grad is not None
    assert torch.isfinite(vd.ad.head.conv.weight.grad).all()

    # 3. Check mask head gradients
    assert vd.md.head.conv.weight.grad is not None
    assert torch.isfinite(vd.md.head.conv.weight.grad).all()

    # 4. Check input gradients
    assert i_loc.grad is not None
    assert torch.isfinite(i_loc.grad).all()
    assert orig_image.grad is not None
    assert torch.isfinite(orig_image.grad).all()
