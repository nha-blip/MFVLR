"""Unit tests for Multi-domain Vision Encoder (MVE), Image Encoder (IE), and Residual Encoder (RE).

PAPER_SPECIFIED constraints tested:
1. Input image I: [2, 3, 224, 224]
2. Local feature I_loc: [2, 1024, 14, 14]
3. Token sequence: 196 spatial + 1 class = 197 tokens of dimension d = 512
4. Exactly B = 4 Image Transformer blocks
5. Global appearance feature I_g: [2, 512] from class token
6. Residual feature I_rg: [2, 512] encoded via SAME Image Encoder
7. Fused visual feature: I_v = I_g + I_rg of shape [2, 512]
8. True IE/RE parameter sharing (identical parameter objects)
9. Gradient flow through I_v to shared Image Encoder
10. Finite output values (no NaN / Inf)
"""

import pytest
import torch
import torch.nn as nn

from models.vision.unet_encoder import UNetEncoder
from models.vision.image_transformer import ImageTransformer, ImageTransformerBlock
from models.vision.image_encoder import ImageEncoder
from models.vision.residual_encoder import ResidualEncoder
from models.vision.mve import MultiDomainVisionEncoder


def test_unet_encoder_output_shape():
    """Verify UNetEncoder outputs [B, 1024, 14, 14] for [B, 3, 224, 224] input."""
    encoder = UNetEncoder(in_channels=3, channels=(128, 256, 512, 1024))
    x = torch.randn(2, 3, 224, 224)
    i_loc = encoder(x, return_skips=False)

    assert i_loc.shape == (2, 1024, 14, 14)
    assert not torch.isnan(i_loc).any()
    assert not torch.isinf(i_loc).any()


def test_unet_encoder_skips():
    """Verify UNetEncoder returns multi-scale skip features."""
    encoder = UNetEncoder(in_channels=3, channels=(128, 256, 512, 1024))
    x = torch.randn(2, 3, 224, 224)
    i_loc, skips = encoder(x, return_skips=True)

    assert i_loc.shape == (2, 1024, 14, 14)
    assert len(skips) == 4
    assert skips[0].shape == (2, 64, 224, 224)
    assert skips[1].shape == (2, 128, 112, 112)
    assert skips[2].shape == (2, 256, 56, 56)
    assert skips[3].shape == (2, 512, 28, 28)


def test_image_transformer_structure():
    """Verify ImageTransformer has exactly B = 4 blocks and preserves shape [B, 197, 512]."""
    transformer = ImageTransformer(
        num_blocks=4,
        embed_dim=512,
        num_heads=8,
        dim_feedforward=2048,
    )
    assert len(transformer.blocks) == 4
    for block in transformer.blocks:
        assert isinstance(block, ImageTransformerBlock)
        assert block.self_attn.num_heads == 8
        assert block.self_attn.embed_dim == 512

    tokens = torch.randn(2, 197, 512)
    out = transformer(tokens)
    assert out.shape == (2, 197, 512)
    assert not torch.isnan(out).any()


def test_image_encoder_forward():
    """Verify ImageEncoder extracts I_loc [2, 1024, 14, 14] and I_g [2, 512]."""
    ie = ImageEncoder(
        in_channels=3,
        local_channels=1024,
        local_height=14,
        local_width=14,
        embed_dim=512,
        num_blocks=4,
        num_heads=8,
        dim_feedforward=2048,
    )
    x = torch.randn(2, 3, 224, 224)
    i_loc, i_g = ie(x)

    # 1. Output shapes
    assert i_loc.shape == (2, 1024, 14, 14)
    assert i_g.shape == (2, 512)

    # 2. Token counts and parameters
    assert ie.cls_token.shape == (1, 1, 512)
    assert ie.pos_embed.shape == (1, 197, 512)
    assert ie.proj.in_features == 1024
    assert ie.proj.out_features == 512

    # 3. Finite outputs
    assert torch.isfinite(i_loc).all()
    assert torch.isfinite(i_g).all()


def test_mve_true_weight_sharing():
    """Verify Residual Encoder (RE) shares exact same parameters and module instance as IE."""
    mve = MultiDomainVisionEncoder(
        in_channels=3,
        local_channels=1024,
        embed_dim=512,
        image_transformer_blocks=4,
    )

    # RE must reference the exact same ImageEncoder object
    assert mve.residual_encoder.image_encoder is mve.image_encoder
    assert mve.re is mve.ie

    # Verify every parameter in RE is identically the parameter in IE
    ie_params = list(mve.image_encoder.parameters())
    re_params = list(mve.residual_encoder.parameters())
    assert len(ie_params) == len(re_params)
    for p_ie, p_re in zip(ie_params, re_params):
        assert p_ie is p_re


def test_mve_full_forward_and_fusion():
    """Verify MVE full forward pass with appearance and residual inputs."""
    mve = MultiDomainVisionEncoder(
        in_channels=3,
        local_channels=1024,
        embed_dim=512,
        image_transformer_blocks=4,
    )

    image = torch.randn(2, 3, 224, 224)
    residual_image = torch.randn(2, 3, 224, 224).abs()  # Non-negative residual

    outputs = mve(image, residual_image=residual_image, return_skips=True)

    assert "i_loc" in outputs
    assert "i_g" in outputs
    assert "i_rg" in outputs
    assert "i_v" in outputs
    assert "skips" in outputs

    assert outputs["i_loc"].shape == (2, 1024, 14, 14)
    assert outputs["i_g"].shape == (2, 512)
    assert outputs["i_rg"].shape == (2, 512)
    assert outputs["i_v"].shape == (2, 512)

    # Verify fusion: I_v == I_g + I_rg
    expected_iv = outputs["i_g"] + outputs["i_rg"]
    assert torch.allclose(outputs["i_v"], expected_iv, atol=1e-6)

    # Verify finite values
    assert torch.isfinite(outputs["i_loc"]).all()
    assert torch.isfinite(outputs["i_g"]).all()
    assert torch.isfinite(outputs["i_rg"]).all()
    assert torch.isfinite(outputs["i_v"]).all()


def test_mve_gradient_backpropagation():
    """Verify gradients propagate back from a scalar loss on I_v to shared ImageEncoder."""
    mve = MultiDomainVisionEncoder(
        in_channels=3,
        local_channels=1024,
        embed_dim=512,
        image_transformer_blocks=4,
    )

    image = torch.randn(2, 3, 224, 224, requires_grad=True)
    residual_image = torch.randn(2, 3, 224, 224, requires_grad=True)

    outputs = mve(image, residual_image=residual_image)
    i_v = outputs["i_v"]

    # Compute a scalar loss
    loss = i_v.sum()
    loss.backward()

    # Verify gradients exist and are finite for shared parameters
    assert mve.image_encoder.cls_token.grad is not None
    assert torch.isfinite(mve.image_encoder.cls_token.grad).all()

    assert mve.image_encoder.pos_embed.grad is not None
    assert torch.isfinite(mve.image_encoder.pos_embed.grad).all()

    assert mve.image_encoder.proj.weight.grad is not None
    assert torch.isfinite(mve.image_encoder.proj.weight.grad).all()

    # Verify input gradients
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert residual_image.grad is not None
    assert torch.isfinite(residual_image.grad).all()
