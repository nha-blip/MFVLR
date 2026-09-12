"""Tests for all six MFVLR loss functions and total loss (Phase 8).

PAPER_SPECIFIED:
- L_fd: Detection Cross-Entropy Loss (Eq. 26)
- L_lr: Language Reconstruction Loss over vocabulary projection (Eq. 24-25)
- L_ar: Appearance Reconstruction MSE Loss (Eq. 17)
- L_fl: Forgery Localization Pixel-wise Cross-Entropy Loss (Eq. 18)
- L_kl: KL Semantic Alignment Loss with temperature tau=0.5 (Eq. 19)
- L_cmc: Cross-Modal Contrastive Loss with unnormalized dot-product and trainable tau (Eq. 20-23)
- Total Loss: L = lambda_fd*L_fd + lambda_lr*L_lr + lambda_cmc*L_cmc + lambda_fl*L_fl + lambda_ar*L_ar + lambda_kl*L_kl (Eq. 27)
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses import (
    ForgeryDetectionLoss,
    LanguageReconstructionLoss,
    AppearanceReconstructionLoss,
    ForgeryLocalizationLoss,
    KLSemanticAlignmentLoss,
    CrossModalContrastiveLoss,
    MFVLRLoss,
    TotalLossOutput,
)


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def embed_dim():
    return 512


@pytest.fixture
def max_tokens():
    return 308


@pytest.fixture
def vocab_size():
    return 49408


def test_forgery_detection_loss(batch_size):
    """Verify L_fd cross-entropy computation on raw 2-class logits."""
    loss_fn = ForgeryDetectionLoss()

    # Raw logits [B, 2]
    y_pre = torch.randn(batch_size, 2, requires_grad=True)
    y_target = torch.randint(0, 2, (batch_size,), dtype=torch.long)

    loss = loss_fn(y_pre, y_target)

    # 1. Scalar, finite, non-negative
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    # 2. Gradients
    loss.backward()
    assert y_pre.grad is not None
    assert (y_pre.grad.abs() > 0).any()

    # 3. Supports one-hot targets [B, 2]
    y_pre_2 = torch.randn(batch_size, 2)
    y_one_hot = F.one_hot(y_target, num_classes=2).float()
    loss_one_hot = loss_fn(y_pre_2, y_one_hot)
    expected_loss = F.cross_entropy(y_pre_2, y_target)
    assert torch.allclose(loss_one_hot, expected_loss)


def test_language_reconstruction_loss():
    """Verify L_lr token cross-entropy computation over vocabulary projection."""
    loss_fn = LanguageReconstructionLoss()

    # Use smaller batch/tokens for isolated test to manage memory
    B, n, s = 2, 20, 1000
    t_pre = torch.randn(B, n, s, requires_grad=True)
    target_tokens = torch.randint(0, s, (B, n), dtype=torch.long)

    loss = loss_fn(t_pre, target_tokens)

    # 1. Scalar, finite, non-negative
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    # 2. Gradients reach logits
    loss.backward()
    assert t_pre.grad is not None
    assert (t_pre.grad.abs() > 0).any()


def test_appearance_reconstruction_loss(batch_size):
    """Verify L_ar mean squared error between input and reconstructed appearance images."""
    loss_fn = AppearanceReconstructionLoss()

    image = torch.rand(batch_size, 3, 224, 224)
    i_pre = image.clone().detach().requires_grad_(True)

    # 1. Exact reconstruction yields zero loss
    perfect_loss = loss_fn(i_pre, image)
    assert torch.allclose(perfect_loss, torch.tensor(0.0), atol=1e-7)

    # 2. Distorted reconstruction yields positive loss
    noisy_pre = (image + 0.1).clamp(0.0, 1.0).requires_grad_(True)
    noisy_loss = loss_fn(noisy_pre, image)
    assert noisy_loss.item() > 0.0

    # 3. Gradients
    noisy_loss.backward()
    assert noisy_pre.grad is not None
    assert (noisy_pre.grad.abs() > 0).any()


def test_appearance_reconstruction_loss_is_strictly_mse_not_l1():
    """Regression test: Explicitly verify AppearanceReconstructionLoss computes Eq. 17 MSE and fails for L1.

    PAPER_SPECIFIED:
    - Eq. (17): L_ar = (1/b) * sum_u (I^u - I_pre^u)^2
    """
    loss_fn = AppearanceReconstructionLoss()

    # Construct deterministic tensors with error magnitude 0.5 where MSE != L1:
    # MSE = (0.5)^2 = 0.25
    # L1  = |0.5|   = 0.50
    image = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    i_pre = torch.full((2, 3, 224, 224), 0.5, dtype=torch.float32)

    loss = loss_fn(i_pre, image)

    # 1. Matches exact MSE mathematical formula
    assert torch.allclose(loss, torch.tensor(0.25), atol=1e-6)

    # 2. Distinct from L1
    assert not torch.allclose(loss, torch.tensor(0.50), atol=1e-2)



def test_forgery_localization_loss(batch_size):
    """Verify L_fl pixel-level 2-class cross-entropy on predicted mask logits."""
    loss_fn = ForgeryLocalizationLoss()

    m_pre = torch.randn(batch_size, 2, 224, 224, requires_grad=True)
    target_mask = torch.randint(0, 2, (batch_size, 224, 224), dtype=torch.long)

    loss = loss_fn(m_pre, target_mask)

    # 1. Scalar, finite, non-negative
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    # 2. Gradients reach mask logits
    loss.backward()
    assert m_pre.grad is not None
    assert (m_pre.grad.abs() > 0).any()

    # 3. Also supports 4D target mask [B, 1, 224, 224]
    m_pre_2 = torch.randn(batch_size, 2, 224, 224)
    target_mask_4d = target_mask.unsqueeze(1)
    loss_4d = loss_fn(m_pre_2, target_mask_4d)
    expected_loss = F.cross_entropy(m_pre_2, target_mask)
    assert torch.allclose(loss_4d, expected_loss)


def test_kl_semantic_alignment_loss(batch_size, embed_dim):
    """PAPER_SPECIFIED test: Verify L_kl exact direction D_KL(P(T_l) || Q(T_lpre)) and tau=0.5."""
    loss_fn = KLSemanticAlignmentLoss(temperature=0.5)

    # Synthetic features
    t_l = torch.randn(batch_size, embed_dim, requires_grad=True)
    t_lpre = torch.randn(batch_size, embed_dim, requires_grad=True)

    # 1. Zero loss when identical
    identical_loss = loss_fn(t_l, t_l)
    assert torch.allclose(identical_loss, torch.tensor(0.0), atol=1e-6)

    # 2. Non-negative loss
    loss = loss_fn(t_l, t_lpre)
    assert loss.item() >= -1e-6

    # 3. Exact manual mathematical formula verification:
    # P = softmax(T_l / 0.5), Q = softmax(T_lpre / 0.5)
    # D_KL(P || Q) = sum P * (log P - log Q)
    with torch.no_grad():
        P = F.softmax(t_l / 0.5, dim=-1)
        log_P = F.log_softmax(t_l / 0.5, dim=-1)
        log_Q = F.log_softmax(t_lpre / 0.5, dim=-1)
        manual_kl = (P * (log_P - log_Q)).sum(dim=-1).mean()

    assert torch.allclose(loss, manual_kl, atol=1e-5), "L_kl MUST strictly equal D_KL(P(T_l) || Q(T_lpre))"

    # 4. Asymmetric direction check: D_KL(P || Q) != D_KL(Q || P)
    reversed_loss = loss_fn(t_lpre, t_l)
    assert not torch.allclose(loss, reversed_loss), "KL divergence is asymmetric: D_KL(P || Q) != D_KL(Q || P)"

    # 5. Gradients reach both T_l and T_lpre
    loss.backward()
    assert t_l.grad is not None and (t_l.grad.abs() > 0).any()
    assert t_lpre.grad is not None and (t_lpre.grad.abs() > 0).any()


def test_cross_modal_contrastive_loss(batch_size, embed_dim):
    """PAPER_SPECIFIED test: Verify L_cmc unnormalized dot product, trainable tau=0.07, and bidirectional CE."""
    loss_fn = CrossModalContrastiveLoss(initial_temperature=0.07, trainable_temperature=True)

    # 1. Initial temperature must be 0.07
    assert math.isclose(loss_fn.tau.item(), 0.07, rel_tol=1e-4)

    # 2. Dot-product similarity matrix check
    i_v = torch.randn(batch_size, embed_dim, requires_grad=True)
    t_l = torch.randn(batch_size, embed_dim, requires_grad=True)

    loss, S = loss_fn(i_v, t_l, return_similarity=True)

    expected_S = torch.matmul(i_v, t_l.t())
    assert torch.allclose(S, expected_S), "CMC similarity MUST be strictly unnormalized dot product (no L2 norm)"

    # 3. Alignment discriminability test:
    # Well-aligned pairs (diagonal high) must have strictly lower loss than mismatched pairs
    aligned_i_v = torch.eye(batch_size, embed_dim)
    aligned_t_l = torch.eye(batch_size, embed_dim)
    aligned_loss = loss_fn(aligned_i_v, aligned_t_l)

    # Shuffle text features to create mismatch
    shuffled_t_l = aligned_t_l[torch.randperm(batch_size)]
    shuffled_loss = loss_fn(aligned_i_v, shuffled_t_l)

    assert aligned_loss.item() < shuffled_loss.item(), "Aligned pairs must achieve lower CMC loss than mismatched pairs"

    # 4. Gradients reach I_v, T_l, and trainable temperature log_tau
    loss.backward()
    assert i_v.grad is not None and (i_v.grad.abs() > 0).any()
    assert t_l.grad is not None and (t_l.grad.abs() > 0).any()
    assert loss_fn.log_tau.grad is not None and (loss_fn.log_tau.grad.abs() > 0).any()


def test_mfvlr_total_loss(batch_size, embed_dim):
    """Verify total weighted loss equals sum of all six components and backpropagates end-to-end."""
    total_loss_fn = MFVLRLoss(
        lambda_fd=1.0,
        lambda_lr=1.0,
        lambda_cmc=1.0,
        lambda_fl=1.0,
        lambda_ar=1.0,
        lambda_kl=1.0,
    )

    # Dummy batch data
    B = batch_size
    y_pre = torch.randn(B, 2, requires_grad=True)
    y_target = torch.randint(0, 2, (B,), dtype=torch.long)

    # Sub-sampled vocab for fast memory-safe testing
    s = 1000
    t_pre = torch.randn(B, 20, s, requires_grad=True)
    target_token_ids = torch.randint(0, s, (B, 20), dtype=torch.long)

    i_v = torch.randn(B, embed_dim, requires_grad=True)
    t_l = torch.randn(B, embed_dim, requires_grad=True)
    t_lpre = torch.randn(B, embed_dim, requires_grad=True)

    m_pre = torch.randn(B, 2, 224, 224, requires_grad=True)
    target_mask = torch.randint(0, 2, (B, 224, 224), dtype=torch.long)

    image = torch.rand(B, 3, 224, 224)
    i_pre = torch.rand(B, 3, 224, 224, requires_grad=True)

    out = total_loss_fn(
        y_pre=y_pre,
        y_target=y_target,
        t_pre=t_pre,
        target_token_ids=target_token_ids,
        i_v=i_v,
        t_l=t_l,
        m_pre=m_pre,
        target_mask=target_mask,
        i_pre=i_pre,
        image=image,
        t_lpre=t_lpre,
    )

    # 1. Check container
    assert isinstance(out, TotalLossOutput)
    assert "total_loss" in out
    assert "loss_fd" in out
    assert "loss_lr" in out
    assert "loss_cmc" in out
    assert "loss_fl" in out
    assert "loss_ar" in out
    assert "loss_kl" in out

    # 2. Check numerical sum
    expected_sum = (
        out.loss_fd
        + out.loss_lr
        + out.loss_cmc
        + out.loss_fl
        + out.loss_ar
        + out.loss_kl
    )
    assert torch.allclose(out.total_loss, expected_sum), "Total loss must equal exact weighted sum of all 6 components"

    # 3. Gradients reach all components
    out.total_loss.backward()

    assert y_pre.grad is not None
    assert t_pre.grad is not None
    assert i_v.grad is not None
    assert t_l.grad is not None
    assert t_lpre.grad is not None
    assert m_pre.grad is not None
    assert i_pre.grad is not None
