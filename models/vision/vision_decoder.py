"""Vision Decoder (VD) containing Appearance Decoder (AD) and Mask Decoder (MD).

PAPER_SPECIFIED:
- Takes local appearance feature I_loc in R^(B x 1024 x 14 x 14)
- Produces predicted appearance image I_pre in R^(B x 3 x 224 x 224) via Appearance Decoder (AD)
- Produces predicted mask logits M_pre in R^(B x 2 x 224 x 224) via Mask Decoder (MD)
- AD and MD share the exact SAME U-Net decoder network and weights.
- Computes appearance residual:
    I_r = |I_pre - I| in R^(B x 3 x 224 x 224)
"""

from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn

from models.vision.unet_decoder import UNetDecoderTrunk
from models.vision.appearance_decoder import AppearanceDecoder
from models.vision.mask_decoder import MaskDecoder


class VisionDecoder(nn.Module):
    """Vision Decoder (VD) combining shared U-Net decoder with AD and MD heads."""

    def __init__(
        self,
        in_channels: int = 1024,
        out_image_channels: int = 3,
        num_mask_classes: int = 2,
        decoder_channels: Sequence[int] = (512, 256, 128, 64),
        skip_channels: Sequence[int] = (512, 256, 128, 64),
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        # 1. Single shared U-Net Decoder Trunk
        self.decoder_trunk = UNetDecoderTrunk(
            in_channels=in_channels,
            channels=decoder_channels,
            skip_channels=skip_channels,
            num_groups=num_groups,
            dropout=dropout,
        )

        # 2. Appearance Decoder using the shared trunk
        self.appearance_decoder = AppearanceDecoder(
            decoder_trunk=self.decoder_trunk,
            out_channels=out_image_channels,
        )

        # 3. Mask Decoder using the EXACT SAME shared trunk
        self.mask_decoder = MaskDecoder(
            decoder_trunk=self.decoder_trunk,
            num_classes=num_mask_classes,
        )

    @property
    def ad(self) -> AppearanceDecoder:
        """Alias for Appearance Decoder."""
        return self.appearance_decoder

    @property
    def md(self) -> MaskDecoder:
        """Alias for Mask Decoder."""
        return self.mask_decoder

    def decode_appearance(
        self,
        i_loc: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Decode I_loc to reconstructed appearance image I_pre."""
        return self.appearance_decoder(i_loc, skips=skips)

    def decode_mask(
        self,
        i_loc: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Decode I_loc to predicted mask logits M_pre."""
        return self.mask_decoder(i_loc, skips=skips)

    @staticmethod
    def compute_residual(
        i_pre: torch.Tensor,
        i_orig: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pixel-wise appearance residual I_r = |I_pre - I|.

        # PAPER_SPECIFIED:
        # I_r = |I_pre - I| in R^(3 x 224 x 224) (Section III-B, III-C, residual generation)
        """
        return torch.abs(i_pre - i_orig)

    def forward(
        self,
        i_loc: torch.Tensor,
        orig_image: Optional[torch.Tensor] = None,
        skips: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run Vision Decoder forward pass for both appearance and mask reconstruction.

        Args:
            i_loc: Local appearance feature tensor [B, 1024, 14, 14]
            orig_image: Optional original appearance image I [B, 3, 224, 224] for residual calculation
            skips: Optional list of multi-scale encoder skip connections

        Returns:
            Dictionary containing:
                "i_pre": Reconstructed appearance image [B, 3, 224, 224]
                "m_pre": Predicted localization mask logits [B, 2, 224, 224]
                "i_r": Appearance residual image [B, 3, 224, 224] (if orig_image provided)
        """
        # Shared feature decoding: compute trunk features once for efficiency
        shared_features = self.decoder_trunk(i_loc, skips=skips)

        # Separate heads applied to shared features
        i_pre = self.appearance_decoder.head(shared_features)
        m_pre = self.mask_decoder.head(shared_features)

        outputs = {
            "i_pre": i_pre,
            "m_pre": m_pre,
        }

        if orig_image is not None:
            i_r = self.compute_residual(i_pre, orig_image)
            outputs["i_r"] = i_r

        return outputs
