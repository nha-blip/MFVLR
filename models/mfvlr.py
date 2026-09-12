"""MFVLR: Multi-domain Fine-grained Vision-Language Reconstruction Network.

PAPER_SPECIFIED:
- Composes:
    1. Multi-domain Vision Encoder (MVE): Image Encoder (IE) + shared Residual Encoder (RE).
    2. Vision Decoder (VD): Shared U-Net decoder trunk + Appearance Decoder (AD) + Mask Decoder (MD).
    3. Residual Generation: I_r = |I_pre - I| in R^(3 x 224 x 224).
    4. Vision Fusion: I_v = I_g + I_rg in R^(B x 512).
    5. Detection Head: Linear(512, 2) producing raw logits y_pre in R^(B x 2).
    6. Feature Adapter: Linear(512, 512) producing predicted language feature T_lpre in R^(B x 512).
    7. Fine-grained Language Transformer (FLT): Language Encoder (E=12) + Language Decoder (D=7) with VIM injection and tied W_voc^T.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from models.vision.mve import MultiDomainVisionEncoder
from models.vision.vision_decoder import VisionDecoder
from models.language.flt import FineGrainedLanguageTransformer
from models.heads.adapter import Adapter
from models.heads.detection_head import DetectionHead


class MFVLROutput(dict):
    """Structured output container for full MFVLR forward pass.

    Provides attribute access and dictionary key access for all intermediate and final tensors.
    """

    def __init__(
        self,
        i_loc: torch.Tensor,
        i_g: torch.Tensor,
        i_pre: torch.Tensor,
        m_pre: torch.Tensor,
        i_r: torch.Tensor,
        i_rg: torch.Tensor,
        i_v: torch.Tensor,
        y_pre: torch.Tensor,
        t_lpre: torch.Tensor,
        t_low: torch.Tensor,
        t_hig: torch.Tensor,
        t_l: torch.Tensor,
        t_rec: torch.Tensor,
        t_pre: Optional[torch.Tensor] = None,
        skips: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__(
            i_loc=i_loc,
            i_g=i_g,
            i_pre=i_pre,
            m_pre=m_pre,
            i_r=i_r,
            i_rg=i_rg,
            i_v=i_v,
            y_pre=y_pre,
            t_lpre=t_lpre,
            t_low=t_low,
            t_hig=t_hig,
            t_l=t_l,
            t_rec=t_rec,
            t_pre=t_pre,
            **kwargs,
        )
        self.i_loc = i_loc
        self.i_g = i_g
        self.i_pre = i_pre
        self.m_pre = m_pre
        self.i_r = i_r
        self.i_rg = i_rg
        self.i_v = i_v
        self.y_pre = y_pre
        self.t_lpre = t_lpre
        self.t_low = t_low
        self.t_hig = t_hig
        self.t_l = t_l
        self.t_rec = t_rec
        self.t_pre = t_pre
        self.skips = skips

    def predict_class(self) -> torch.Tensor:
        """Return predicted class index [B] via argmax over detection logits y_pre."""
        return torch.argmax(self.y_pre, dim=1)

    def predict_fake_prob(self, fake_class_index: int = 1) -> torch.Tensor:
        """Return predicted probability of fake class [B] via softmax over y_pre."""
        return torch.softmax(self.y_pre, dim=1)[:, fake_class_index]

    def predict_mask(self) -> torch.Tensor:
        """Return predicted binary manipulation mask [B, 224, 224] via argmax over mask logits m_pre."""
        return torch.argmax(self.m_pre, dim=1)


class MFVLRInferenceOutput(dict):
    """Structured output container for image-only MFVLR inference pass.

    PAPER_SPECIFIED:
    - Image-only inference produces 2-class detection logits y_pre and mask logits m_pre.
    - No language tokens, text prompts, or tokenizer are involved.
    """

    def __init__(
        self,
        y_pre: torch.Tensor,
        m_pre: torch.Tensor,
        i_pre: Optional[torch.Tensor] = None,
        i_r: Optional[torch.Tensor] = None,
        i_v: Optional[torch.Tensor] = None,
        i_g: Optional[torch.Tensor] = None,
        i_rg: Optional[torch.Tensor] = None,
        i_loc: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(
            y_pre=y_pre,
            m_pre=m_pre,
            i_pre=i_pre,
            i_r=i_r,
            i_v=i_v,
            i_g=i_g,
            i_rg=i_rg,
            i_loc=i_loc,
            **kwargs,
        )
        self.y_pre = y_pre
        self.m_pre = m_pre
        self.i_pre = i_pre
        self.i_r = i_r
        self.i_v = i_v
        self.i_g = i_g
        self.i_rg = i_rg
        self.i_loc = i_loc

    def predict_class(self) -> torch.Tensor:
        """Return predicted class index [B] via argmax over detection logits y_pre."""
        return torch.argmax(self.y_pre, dim=1)

    def predict_fake_prob(self, fake_class_index: int = 1) -> torch.Tensor:
        """Return predicted probability of fake class [B] via softmax over y_pre."""
        return torch.softmax(self.y_pre, dim=1)[:, fake_class_index]

    def predict_mask(self) -> torch.Tensor:
        """Return predicted binary manipulation mask [B, 224, 224] via argmax over mask logits m_pre."""
        return torch.argmax(self.m_pre, dim=1)


class MFVLR(nn.Module):
    """Complete Multi-domain Fine-grained Vision-Language Reconstruction Model."""

    def __init__(
        self,
        # Image dimensions & channels
        in_channels: int = 3,
        local_channels: int = 1024,
        local_height: int = 14,
        local_width: int = 14,
        embed_dim: int = 512,
        num_classes: int = 2,
        mask_channels: int = 2,
        # Vision backbone architecture
        image_transformer_blocks: int = 4,
        image_attention_heads: int = 8,
        unet_encoder_channels: Sequence[int] = (128, 256, 512, 1024),
        unet_decoder_channels: Sequence[int] = (512, 256, 128, 64),
        # Language architecture
        vocab_size: int = 49408,
        max_text_tokens: int = 308,
        language_encoder_blocks: int = 12,
        language_decoder_blocks: int = 7,
        language_attention_heads: int = 8,
        vim_heads: int = 8,
        dim_feedforward: int = 2048,
        # Regularization & initialization
        dropout: float = 0.0,
        activation: str = "gelu",
        pos_init_std: float = 0.02,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_text_tokens = max_text_tokens
        self.num_classes = num_classes

        # 1. Multi-domain Vision Encoder (MVE)
        # Manages ImageEncoder and shared ResidualEncoder instance
        self.mve = MultiDomainVisionEncoder(
            in_channels=in_channels,
            local_channels=local_channels,
            local_height=local_height,
            local_width=local_width,
            embed_dim=embed_dim,
            image_transformer_blocks=image_transformer_blocks,
            image_attention_heads=image_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            unet_channels=unet_encoder_channels,
        )

        # 2. Vision Decoder (VD)
        # Manages shared U-Net decoder trunk, AppearanceDecoder, and MaskDecoder
        self.vision_decoder = VisionDecoder(
            in_channels=local_channels,
            out_image_channels=in_channels,
            num_mask_classes=mask_channels,
            decoder_channels=unet_decoder_channels,
            dropout=dropout,
        )

        # 3. Fine-grained Language Transformer (FLT)
        # Integrates LE (E=12) and LD (D=7) with tied W_voc^T
        self.flt = FineGrainedLanguageTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_text_tokens=max_text_tokens,
            encoder_blocks=language_encoder_blocks,
            decoder_blocks=language_decoder_blocks,
            num_heads=language_attention_heads,
            vim_heads=vim_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            pos_init_std=pos_init_std,
        )

        # 4. Feature Adapter (I_v -> T_lpre)
        self.adapter = Adapter(
            in_dim=embed_dim,
            out_dim=embed_dim,
        )

        # 5. Detection Classification Head (I_v -> y_pre)
        self.detection_head = DetectionHead(
            in_dim=embed_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        image: torch.Tensor,
        token_ids: torch.Tensor,
        return_logits: bool = True,
    ) -> MFVLROutput:
        """Execute full end-to-end MFVLR training forward pass.

        Args:
            image: Appearance input image I [B, 3, 224, 224] in [0, 1]
            token_ids: Prompt token IDs [B, 308] in [0, 49407]
            return_logits: Whether to compute language reconstruction logits T_pre [B, 308, 49408].

        Returns:
            MFVLROutput containing all 14 intermediate and output representations.
        """
        # Step 1: MVE encodes appearance image I -> I_loc [B, 1024, 14, 14], I_g [B, 512], skips
        i_loc, i_g, skips = self.mve.encode_image(image, return_skips=True)

        # Step 2: Vision Decoder reconstructs appearance and predicts mask
        # I_pre in R^(B x 3 x 224 x 224), M_pre in R^(B x 2 x 224 x 224)
        vd_out = self.vision_decoder(i_loc, orig_image=image, skips=skips)
        i_pre = vd_out["i_pre"]
        m_pre = vd_out["m_pre"]
        i_r = vd_out["i_r"]  # I_r = |I_pre - I| in R^(B x 3 x 224 x 224)

        # Step 3: Shared Residual Encoder encodes I_r -> I_rg [B, 512]
        i_rg = self.mve.encode_residual(i_r)

        # Step 4: Multi-domain visual feature fusion: I_v = I_g + I_rg in R^(B x 512)
        i_v = self.mve.fuse(i_g, i_rg)

        # Step 5: Detection Head classifies I_v -> y_pre in R^(B x 2)
        y_pre = self.detection_head(i_v)

        # Step 6: Feature Adapter projects I_v -> T_lpre in R^(B x 512)
        t_lpre = self.adapter(i_v)

        # Step 7: Fine-grained Language Transformer encodes prompt conditioned on I_v
        # and reconstructs tokens via tied vocabulary projection W_voc^T
        flt_out = self.flt(token_ids, i_v, return_logits=return_logits)

        return MFVLROutput(
            i_loc=i_loc,
            i_g=i_g,
            i_pre=i_pre,
            m_pre=m_pre,
            i_r=i_r,
            i_rg=i_rg,
            i_v=i_v,
            y_pre=y_pre,
            t_lpre=t_lpre,
            t_low=flt_out.t_low,
            t_hig=flt_out.t_hig,
            t_l=flt_out.t_l,
            t_rec=flt_out.t_rec,
            t_pre=flt_out.t_pre,
            skips=skips,
        )

    def forward_image_only(
        self,
        image: torch.Tensor,
        return_intermediates: bool = False,
    ) -> MFVLRInferenceOutput:
        """Execute paper-specified image-only inference forward pass.

        PAPER_SPECIFIED:
        - Image-only inference does NOT require or execute:
            * prompt / token_ids / tokenizer
            * Language Encoder (LE)
            * Language Decoder (LD)
            * Fine-grained Language Transformer (FLT)
            * Feature Adapter
            * Loss functions (CMC, KL, etc.)
        - Inference graph:
            I -> IE -> I_loc, I_g
              -> VD -> I_pre, M_pre
              -> I_r = |I_pre - I|
              -> RE -> I_rg
              -> I_v = I_g + I_rg
              -> Detection Head -> y_pre

        Args:
            image: Input image I [B, 3, 224, 224] in [0, 1]
            return_intermediates: Whether to include intermediate representations in output.

        Returns:
            MFVLRInferenceOutput containing y_pre, m_pre (and intermediates if requested).
        """
        # Step 1: MVE encodes appearance image I -> I_loc [B, 1024, 14, 14], I_g [B, 512], skips
        i_loc, i_g, skips = self.mve.encode_image(image, return_skips=True)

        # Step 2: Vision Decoder reconstructs appearance and predicts mask
        # I_pre in R^(B x 3 x 224 x 224), M_pre in R^(B x 2 x 224 x 224)
        vd_out = self.vision_decoder(i_loc, orig_image=image, skips=skips)
        i_pre = vd_out["i_pre"]
        m_pre = vd_out["m_pre"]
        i_r = vd_out["i_r"]  # I_r = |I_pre - I| in R^(B x 3 x 224 x 224)

        # Step 3: Shared Residual Encoder encodes I_r -> I_rg [B, 512]
        i_rg = self.mve.encode_residual(i_r)

        # Step 4: Multi-domain visual feature fusion: I_v = I_g + I_rg in R^(B x 512)
        i_v = self.mve.fuse(i_g, i_rg)

        # Step 5: Detection Head classifies I_v -> y_pre in R^(B x 2)
        y_pre = self.detection_head(i_v)

        return MFVLRInferenceOutput(
            y_pre=y_pre,
            m_pre=m_pre,
            i_pre=i_pre if return_intermediates else None,
            i_r=i_r if return_intermediates else None,
            i_v=i_v if return_intermediates else None,
            i_g=i_g if return_intermediates else None,
            i_rg=i_rg if return_intermediates else None,
            i_loc=i_loc if return_intermediates else None,
        )

    def count_parameters(self) -> Dict[str, int]:
        """Compute unique parameter counts across submodules and entire model."""
        return {
            "mve": sum(p.numel() for p in set(self.mve.parameters())),
            "vision_decoder": sum(p.numel() for p in set(self.vision_decoder.parameters())),
            "flt": sum(p.numel() for p in set(self.flt.parameters())),
            "adapter": sum(p.numel() for p in set(self.adapter.parameters())),
            "detection_head": sum(p.numel() for p in set(self.detection_head.parameters())),
            "total_unique": sum(p.numel() for p in set(self.parameters())),
            "trainable_unique": sum(p.numel() for p in set(self.parameters()) if p.requires_grad),
        }

