"""End-to-End Integration Tests for Full MFVLR Architecture (Phase 9).

PAPER_SPECIFIED:
- Composes MVE, Vision Decoder, FLT, Adapter, Detection Head, and MFVLRLoss.
- Preserves IE/RE weight sharing, AD/MD decoder trunk sharing, and FLT tied W_voc^T.
- Enforces strict execution order: I -> I_loc, I_g -> VD(I_loc) -> I_pre, M_pre -> I_r = |I_pre - I| -> RE(I_r) -> I_rg -> I_v = I_g + I_rg -> (Adapter, DetectionHead, FLT(token_ids, I_v)).
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mfvlr import MFVLR, MFVLROutput
from models.losses import MFVLRLoss, TotalLossOutput


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def dummy_image(batch_size):
    """Synthetic image I [B, 3, 224, 224] in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(batch_size, 3, 224, 224)


@pytest.fixture
def dummy_token_ids(batch_size):
    """Synthetic prompt token IDs [B, 308] in [0, 49407]."""
    torch.manual_seed(43)
    return torch.randint(0, 49408, (batch_size, 308), dtype=torch.long)


@pytest.fixture
def dummy_detection_target(batch_size):
    """Synthetic classification labels [B] in {0, 1}."""
    torch.manual_seed(44)
    return torch.randint(0, 2, (batch_size,), dtype=torch.long)


@pytest.fixture
def dummy_mask_target(batch_size):
    """Synthetic localization mask [B, 224, 224] in {0, 1}."""
    torch.manual_seed(45)
    return torch.randint(0, 2, (batch_size, 224, 224), dtype=torch.long)


def test_mfvlr_weight_sharing_and_tying():
    """Verify all required weight sharing and tying properties are intact after full model construction."""
    model = MFVLR()

    # 1. IE / RE sharing: ResidualEncoder wraps the exact same ImageEncoder instance
    assert model.mve.re is model.mve.ie
    assert model.mve.image_encoder is model.mve.residual_encoder.image_encoder
    assert (
        model.mve.image_encoder.unet_encoder.init_conv.conv1.weight
        is model.mve.residual_encoder.image_encoder.unet_encoder.init_conv.conv1.weight
    )
    assert (
        model.mve.image_encoder.unet_encoder.init_conv.conv1.weight.data_ptr()
        == model.mve.residual_encoder.image_encoder.unet_encoder.init_conv.conv1.weight.data_ptr()
    )

    # 2. AD / MD sharing: AppearanceDecoder and MaskDecoder share the exact same U-Net decoder trunk
    assert model.vision_decoder.appearance_decoder.decoder_trunk is model.vision_decoder.mask_decoder.decoder_trunk
    assert model.vision_decoder.decoder_trunk is model.vision_decoder.appearance_decoder.decoder_trunk
    assert (
        model.vision_decoder.appearance_decoder.decoder_trunk.up1.conv.conv1.weight.data_ptr()
        == model.vision_decoder.mask_decoder.decoder_trunk.up1.conv.conv1.weight.data_ptr()
    )

    # 3. FLT vocabulary matrix tying: LanguageDecoder shares the exact same W_voc parameter as LanguageEncoder
    assert model.flt.decoder.token_embedding.weight is model.flt.encoder.embeddings.token_embed.weight
    assert (
        model.flt.decoder.token_embedding.weight.data_ptr()
        == model.flt.encoder.embeddings.token_embed.weight.data_ptr()
    )


def test_mfvlr_forward_shapes_and_finiteness(dummy_image, dummy_token_ids, batch_size):
    """Verify full end-to-end forward pass shapes and finiteness across all 14 representations."""
    model = MFVLR()

    out = model(dummy_image, dummy_token_ids, return_logits=True)

    # 1. Container type
    assert isinstance(out, MFVLROutput)

    # 2. Exact paper shapes
    assert out.i_loc.shape == (batch_size, 1024, 14, 14)
    assert out.i_g.shape == (batch_size, 512)
    assert out.i_pre.shape == (batch_size, 3, 224, 224)
    assert out.m_pre.shape == (batch_size, 2, 224, 224)
    assert out.i_r.shape == (batch_size, 3, 224, 224)
    assert out.i_rg.shape == (batch_size, 512)
    assert out.i_v.shape == (batch_size, 512)
    assert out.y_pre.shape == (batch_size, 2)
    assert out.t_lpre.shape == (batch_size, 512)
    assert out.t_low.shape == (batch_size, 308, 512)
    assert out.t_hig.shape == (batch_size, 308, 512)
    assert out.t_l.shape == (batch_size, 512)
    assert out.t_rec.shape == (batch_size, 308, 512)
    assert out.t_pre.shape == (batch_size, 308, 49408)

    # 3. Finiteness
    assert torch.isfinite(out.i_loc).all()
    assert torch.isfinite(out.i_g).all()
    assert torch.isfinite(out.i_pre).all()
    assert torch.isfinite(out.m_pre).all()
    assert torch.isfinite(out.i_r).all()
    assert torch.isfinite(out.i_rg).all()
    assert torch.isfinite(out.i_v).all()
    assert torch.isfinite(out.y_pre).all()
    assert torch.isfinite(out.t_lpre).all()
    assert torch.isfinite(out.t_low).all()
    assert torch.isfinite(out.t_hig).all()
    assert torch.isfinite(out.t_l).all()
    assert torch.isfinite(out.t_rec).all()
    assert torch.isfinite(out.t_pre).all()


def test_mfvlr_residual_and_fusion_exact_equality(dummy_image, dummy_token_ids):
    """Verify exact mathematical relations: I_r = |I_pre - I| and I_v = I_g + I_rg."""
    model = MFVLR()

    out = model(dummy_image, dummy_token_ids, return_logits=False)

    # 1. Residual equality: I_r == |I_pre - I|
    expected_residual = torch.abs(out.i_pre - dummy_image)
    assert torch.allclose(out.i_r, expected_residual, atol=1e-6), "I_r must strictly equal |I_pre - I|"

    # 2. Visual fusion equality: I_v == I_g + I_rg
    expected_fused = out.i_g + out.i_rg
    assert torch.allclose(out.i_v, expected_fused, atol=1e-6), "I_v must strictly equal I_g + I_rg"

    # 3. Last token equality: T_l == T_hig[:, -1, :]
    assert torch.equal(out.t_l, out.t_hig[:, -1, :]), "T_l must strictly equal T_hig[:, -1, :]"


def test_mfvlr_end_to_end_loss_and_backward(
    dummy_image,
    dummy_token_ids,
    dummy_detection_target,
    dummy_mask_target,
):
    """Verify complete forward and backward pass through full MFVLR model and MFVLRLoss."""
    model = MFVLR()
    loss_fn = MFVLRLoss()

    # Enable gradients on model
    model.train()

    # Forward pass
    out = model(dummy_image, dummy_token_ids, return_logits=True)

    # Compute all six losses and total loss
    losses = loss_fn(
        y_pre=out.y_pre,
        y_target=dummy_detection_target,
        t_pre=out.t_pre,
        target_token_ids=dummy_token_ids,
        i_v=out.i_v,
        t_l=out.t_l,
        m_pre=out.m_pre,
        target_mask=dummy_mask_target,
        i_pre=out.i_pre,
        image=dummy_image,
        t_lpre=out.t_lpre,
    )

    # 1. Inspect scalar losses
    assert torch.isfinite(losses.total_loss)
    assert torch.isfinite(losses.loss_fd)
    assert torch.isfinite(losses.loss_lr)
    assert torch.isfinite(losses.loss_cmc)
    assert torch.isfinite(losses.loss_fl)
    assert torch.isfinite(losses.loss_ar)
    assert torch.isfinite(losses.loss_kl)

    # 2. Verify total loss equals exact sum of 6 components
    expected_sum = (
        losses.loss_fd
        + losses.loss_lr
        + losses.loss_cmc
        + losses.loss_fl
        + losses.loss_ar
        + losses.loss_kl
    )
    assert torch.allclose(losses.total_loss, expected_sum, atol=1e-5)

    # 3. Execute backward pass
    losses.total_loss.backward()

    # 4. Comprehensive gradient audit across ALL submodules:

    # A. Vision Encoder (IE/RE shared)
    assert model.mve.image_encoder.unet_encoder.init_conv.conv1.weight.grad is not None
    assert (model.mve.image_encoder.unet_encoder.init_conv.conv1.weight.grad.abs() > 0).any()
    assert model.mve.image_encoder.proj.weight.grad is not None
    assert (model.mve.image_encoder.proj.weight.grad.abs() > 0).any()
    assert model.mve.image_encoder.transformer.blocks[0].self_attn.in_proj_weight.grad is not None
    assert (model.mve.image_encoder.transformer.blocks[0].self_attn.in_proj_weight.grad.abs() > 0).any()

    # B. Vision Decoder (AD/MD shared trunk and heads)
    assert model.vision_decoder.decoder_trunk.up1.conv.conv1.weight.grad is not None
    assert (model.vision_decoder.decoder_trunk.up1.conv.conv1.weight.grad.abs() > 0).any()
    assert model.vision_decoder.appearance_decoder.head.conv.weight.grad is not None
    assert (model.vision_decoder.appearance_decoder.head.conv.weight.grad.abs() > 0).any()
    assert model.vision_decoder.mask_decoder.head.conv.weight.grad is not None
    assert (model.vision_decoder.mask_decoder.head.conv.weight.grad.abs() > 0).any()

    # C. Adapter & Detection Head
    assert model.adapter.proj.weight.grad is not None
    assert (model.adapter.proj.weight.grad.abs() > 0).any()
    assert model.detection_head.fc.weight.grad is not None
    assert (model.detection_head.fc.weight.grad.abs() > 0).any()

    # D. FLT (Token embeddings / tied W_voc, positional embeddings)
    assert model.flt.token_embedding.weight.grad is not None
    assert (model.flt.token_embedding.weight.grad.abs() > 0).any()
    assert model.flt.encoder.embeddings.pos_embed.grad is not None
    assert (model.flt.encoder.embeddings.pos_embed.grad.abs() > 0).any()
    assert model.flt.decoder.pos_embed.grad is not None
    assert (model.flt.decoder.pos_embed.grad.abs() > 0).any()
    assert model.flt.decoder.bos_embed.grad is not None
    assert (model.flt.decoder.bos_embed.grad.abs() > 0).any()

    # E. LE Blocks (Self-Attn, FFN, VIM)
    assert model.flt.encoder.blocks[0].self_attn.in_proj_weight.grad is not None
    assert (model.flt.encoder.blocks[0].self_attn.in_proj_weight.grad.abs() > 0).any()
    assert model.flt.encoder.blocks[0].ffn[0].weight.grad is not None
    assert (model.flt.encoder.blocks[0].ffn[0].weight.grad.abs() > 0).any()
    assert model.flt.encoder.blocks[0].vim.w_val.weight.grad is not None
    assert (model.flt.encoder.blocks[0].vim.w_val.weight.grad.abs() > 0).any()
    assert model.flt.encoder.blocks[0].vim.w_fc.weight.grad is not None
    assert (model.flt.encoder.blocks[0].vim.w_fc.weight.grad.abs() > 0).any()

    # F. LD Blocks (MMHA, Cross-MHA, FFN, VIM)
    assert model.flt.decoder.blocks[0].self_attn.in_proj_weight.grad is not None
    assert (model.flt.decoder.blocks[0].self_attn.in_proj_weight.grad.abs() > 0).any()
    assert model.flt.decoder.blocks[0].cross_attn.in_proj_weight.grad is not None
    assert (model.flt.decoder.blocks[0].cross_attn.in_proj_weight.grad.abs() > 0).any()
    assert model.flt.decoder.blocks[0].ffn[0].weight.grad is not None
    assert (model.flt.decoder.blocks[0].ffn[0].weight.grad.abs() > 0).any()
    assert model.flt.decoder.blocks[0].vim.w_val.weight.grad is not None
    assert (model.flt.decoder.blocks[0].vim.w_val.weight.grad.abs() > 0).any()
    assert model.flt.decoder.blocks[0].vim.w_fc.weight.grad is not None
    assert (model.flt.decoder.blocks[0].vim.w_fc.weight.grad.abs() > 0).any()

    # G. Loss parameter (trainable CMC temperature log_tau)
    assert loss_fn.cmc_loss.log_tau.grad is not None
    assert (loss_fn.cmc_loss.log_tau.grad.abs() > 0).any()


def test_mfvlr_train_eval_modes(dummy_image, dummy_token_ids):
    """Verify standard model.train() and model.eval() execution without modifying computational graph."""
    model = MFVLR()

    # Train mode
    model.train()
    out_train = model(dummy_image, dummy_token_ids, return_logits=False)
    assert out_train.i_v.shape == (2, 512)

    # Eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(dummy_image, dummy_token_ids, return_logits=False)
    assert out_eval.i_v.shape == (2, 512)
    assert torch.isfinite(out_eval.i_v).all()
