"""Tests for Adapter and DetectionHead (Phase 8).

PAPER_SPECIFIED:
- Adapter projects I_v in R^(B x 512) to predicted language feature T_lpre in R^(B x 512).
- DetectionHead maps I_v in R^(B x 512) to 2-class raw detection logits y_pre in R^(B x 2).
"""

import pytest
import torch
import torch.nn as nn

from models.heads import Adapter, DetectionHead


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def embed_dim():
    return 512


@pytest.fixture
def dummy_visual_feature(batch_size, embed_dim):
    """Synthetic visual feature I_v in R^(B x 512)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, embed_dim)


def test_adapter_forward_shapes_and_gradients(dummy_visual_feature, batch_size, embed_dim):
    """Verify Adapter forward pass, 2D/3D visual input support, and gradient flow."""
    adapter = Adapter(in_dim=embed_dim, out_dim=embed_dim)

    # 1. 2D visual feature [B, 512] -> [B, 512]
    i_v_2d = dummy_visual_feature.clone().detach().requires_grad_(True)
    t_lpre_2d = adapter(i_v_2d)

    assert t_lpre_2d.shape == (batch_size, embed_dim)
    assert torch.isfinite(t_lpre_2d).all()

    # 2. 3D visual feature [B, 1, 512] -> [B, 512]
    i_v_3d = dummy_visual_feature.unsqueeze(1)
    t_lpre_3d = adapter(i_v_3d)

    assert t_lpre_3d.shape == (batch_size, embed_dim)
    assert torch.allclose(t_lpre_2d, t_lpre_3d)

    # 3. Gradients
    loss = t_lpre_2d.sum()
    loss.backward()

    assert adapter.proj.weight.grad is not None
    assert (adapter.proj.weight.grad.abs() > 0).any()
    assert i_v_2d.grad is not None
    assert (i_v_2d.grad.abs() > 0).any()


def test_detection_head_forward_shapes_and_gradients(dummy_visual_feature, batch_size, embed_dim):
    """Verify DetectionHead raw logits output, 2D/3D support, and gradient flow."""
    head = DetectionHead(in_dim=embed_dim, num_classes=2)

    # 1. 2D visual feature [B, 512] -> [B, 2]
    i_v_2d = dummy_visual_feature.clone().detach().requires_grad_(True)
    logits_2d = head(i_v_2d)

    assert logits_2d.shape == (batch_size, 2)
    assert torch.isfinite(logits_2d).all()

    # 2. 3D visual feature [B, 1, 512] -> [B, 2]
    i_v_3d = dummy_visual_feature.unsqueeze(1)
    logits_3d = head(i_v_3d)

    assert logits_3d.shape == (batch_size, 2)
    assert torch.allclose(logits_2d, logits_3d)

    # 3. Gradients
    loss = logits_2d.sum()
    loss.backward()

    assert head.fc.weight.grad is not None
    assert (head.fc.weight.grad.abs() > 0).any()
    assert i_v_2d.grad is not None
    assert (i_v_2d.grad.abs() > 0).any()
