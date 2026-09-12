"""Unit tests for Vision Injection Module (VIM) per Eq. (4)-(11).

PAPER_SPECIFIED constraints tested:
1. Language representation: T_tok^j [2, 308, 512]
2. Visual global feature: I_v [2, 1, 512] (and [2, 512])
3. Q/K/V mapping: Language = Query, Vision = Key/Value
4. Head split: 8 heads, head dimension 64 (512 / 8 = 64)
5. Intermediate shapes:
    - Q: [2, 8, 308, 64]
    - K: [2, 8, 1, 64]
    - V: [2, 8, 1, 64]
    - Attention scores / probabilities: [2, 8, 308, 1]
    - Attention probabilities == 1.0 (Singleton K/V property)
    - Concatenated T_glo: [2, 308, 512]
    - Output T_add: [2, 308, 512]
6. Output finite values (no NaN / Inf)
7. Gradient backpropagation to W_que, W_key, W_val, W_fc
"""

import math
import pytest
import torch
import torch.nn as nn

from models.language.vim import VisionInjectionModule, VIM


def test_vim_initialization_and_hyperparameters():
    """Verify VIM dimensions, head partitions, and projection matrices."""
    vim = VisionInjectionModule(embed_dim=512, num_heads=8)

    assert vim.embed_dim == 512
    assert vim.num_heads == 8
    assert vim.head_dim == 64
    assert math.isclose(vim.scale, 1.0 / 8.0, rel_tol=1e-6)

    # 4 trainable linear projection matrices (Eq. 4-6, Eq. 11)
    assert isinstance(vim.w_que, nn.Linear)
    assert vim.w_que.in_features == 512 and vim.w_que.out_features == 512

    assert isinstance(vim.w_key, nn.Linear)
    assert vim.w_key.in_features == 512 and vim.w_key.out_features == 512

    assert isinstance(vim.w_val, nn.Linear)
    assert vim.w_val.in_features == 512 and vim.w_val.out_features == 512

    assert isinstance(vim.w_fc, nn.Linear)
    assert vim.w_fc.in_features == 512 and vim.w_fc.out_features == 512


def test_vim_forward_3d_inputs():
    """Verify VIM forward pass with language [2, 308, 512] and vision [2, 1, 512]."""
    vim = VisionInjectionModule(embed_dim=512, num_heads=8)

    t_tok = torch.randn(2, 308, 512)
    i_v = torch.randn(2, 1, 512)

    t_add, attn = vim(t_tok, i_v, return_attention=True)

    # 1. Output shape
    assert t_add.shape == (2, 308, 512)
    assert torch.isfinite(t_add).all()

    # 2. Attention shape: [B, r, n, 1] -> [2, 8, 308, 1]
    assert attn.shape == (2, 8, 308, 1)
    assert torch.isfinite(attn).all()


def test_vim_forward_2d_vision_input():
    """Verify VIM automatically unsqueezes 2D vision input [2, 512] -> [2, 1, 512]."""
    vim = VisionInjectionModule(embed_dim=512, num_heads=8)

    t_tok = torch.randn(2, 308, 512)
    i_v_2d = torch.randn(2, 512)

    t_add = vim(t_tok, i_v_2d, return_attention=False)
    assert t_add.shape == (2, 308, 512)
    assert torch.isfinite(t_add).all()


def test_vim_singleton_attention_property():
    """Verify that softmax over a length-1 key dimension mathematically produces exactly 1.0."""
    vim = VisionInjectionModule(embed_dim=512, num_heads=8)

    t_tok = torch.randn(2, 308, 512)
    i_v = torch.randn(2, 1, 512)

    _, attn = vim(t_tok, i_v, return_attention=True)

    # Attention shape: [2, 8, 308, 1]
    expected_ones = torch.ones_like(attn)
    assert torch.allclose(attn, expected_ones, atol=1e-6)
    assert (attn >= 0.0).all()
    assert (attn <= 1.0).all()


def test_vim_gradient_backpropagation():
    """Verify gradient propagation through VIM to projection weights.

    MATHEMATICAL CHECK:
    - W_val and W_fc receive direct non-zero gradients.
    - W_que and W_key are connected to the computational graph (grad is not None).
      Because softmax([s]) = 1.0 has derivative s * (1 - s) = 0.0, gradients through
      the singleton softmax path are mathematically zero.
    """
    vim = VisionInjectionModule(embed_dim=512, num_heads=8)

    t_tok = torch.randn(2, 308, 512, requires_grad=True)
    i_v = torch.randn(2, 1, 512, requires_grad=True)

    t_add = vim(t_tok, i_v)
    loss = t_add.sum()
    loss.backward()

    # 1. W_fc and W_val receive non-zero gradients
    assert vim.w_fc.weight.grad is not None
    assert torch.isfinite(vim.w_fc.weight.grad).all()
    assert (vim.w_fc.weight.grad.abs() > 0).any()

    assert vim.w_val.weight.grad is not None
    assert torch.isfinite(vim.w_val.weight.grad).all()
    assert (vim.w_val.weight.grad.abs() > 0).any()

    # 2. W_que and W_key are connected to the backward graph (grad is not None)
    assert vim.w_que.weight.grad is not None
    assert torch.isfinite(vim.w_que.weight.grad).all()

    assert vim.w_key.weight.grad is not None
    assert torch.isfinite(vim.w_key.weight.grad).all()

    # 3. Input gradients
    assert t_tok.grad is not None
    assert torch.isfinite(t_tok.grad).all()
    assert (t_tok.grad.abs() > 0).any()

    assert i_v.grad is not None
    assert torch.isfinite(i_v.grad).all()
    assert (i_v.grad.abs() > 0).any()


def test_vim_step_by_step_shapes():
    """Verify explicit tensor shapes at every step of Eq. (4)-(11)."""
    B, n, d, r = 2, 308, 512, 8
    head_dim = d // r  # 64

    vim = VisionInjectionModule(embed_dim=d, num_heads=r)
    t_tok = torch.randn(B, n, d)
    i_v = torch.randn(B, 1, d)

    # Step 1: Linear projections (Eq. 4-6)
    q = vim.w_que(t_tok)
    k = vim.w_key(i_v)
    v = vim.w_val(i_v)
    assert q.shape == (B, n, d)
    assert k.shape == (B, 1, d)
    assert v.shape == (B, 1, d)

    # Step 2: Head partition (Eq. 7-9)
    Q = q.view(B, n, r, head_dim).transpose(1, 2)
    K = k.view(B, 1, r, head_dim).transpose(1, 2)
    V = v.view(B, 1, r, head_dim).transpose(1, 2)
    assert Q.shape == (B, r, n, head_dim)  # [2, 8, 308, 64]
    assert K.shape == (B, r, 1, head_dim)  # [2, 8, 1, 64]
    assert V.shape == (B, r, 1, head_dim)  # [2, 8, 1, 64]

    # Step 3: Raw attention scores & Softmax (Eq. 10)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    assert scores.shape == (B, r, n, 1)    # [2, 8, 308, 1]

    attn_probs = torch.softmax(scores, dim=-1)
    assert attn_probs.shape == (B, r, n, 1)  # [2, 8, 308, 1]

    t_glo_heads = torch.matmul(attn_probs, V)
    assert t_glo_heads.shape == (B, r, n, head_dim)  # [2, 8, 308, 64]

    # Step 4: Concatenation (Eq. 11)
    t_glo = t_glo_heads.transpose(1, 2).contiguous().view(B, n, d)
    assert t_glo.shape == (B, n, d)        # [2, 308, 512]

    # Step 5: Output projection + residual addition (Eq. 11)
    t_add = vim.w_fc(t_glo) + t_tok
    assert t_add.shape == (B, n, d)        # [2, 308, 512]
