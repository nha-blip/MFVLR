"""Total MFVLR Multi-Task Loss combining all six objectives (Eq. 27).

PAPER_SPECIFIED:
- Equation (27): L = lambda_fd * L_fd + lambda_lr * L_lr + lambda_cmc * L_cmc + lambda_fl * L_fl + lambda_ar * L_ar + lambda_kl * L_kl
- All six weights lambda are 1.0 by default in the paper.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from models.losses.detection_loss import ForgeryDetectionLoss
from models.losses.language_reconstruction_loss import LanguageReconstructionLoss
from models.losses.appearance_reconstruction_loss import AppearanceReconstructionLoss
from models.losses.localization_loss import ForgeryLocalizationLoss
from models.losses.kl_loss import KLSemanticAlignmentLoss
from models.losses.cmc_loss import CrossModalContrastiveLoss


class TotalLossOutput(dict):
    """Container for total loss and component losses."""

    def __init__(
        self,
        total_loss: torch.Tensor,
        loss_fd: torch.Tensor,
        loss_lr: torch.Tensor,
        loss_cmc: torch.Tensor,
        loss_fl: torch.Tensor,
        loss_ar: torch.Tensor,
        loss_kl: torch.Tensor,
        **kwargs,
    ):
        super().__init__(
            total_loss=total_loss,
            loss_fd=loss_fd,
            loss_lr=loss_lr,
            loss_cmc=loss_cmc,
            loss_fl=loss_fl,
            loss_ar=loss_ar,
            loss_kl=loss_kl,
            **kwargs,
        )
        self.total_loss = total_loss
        self.loss_fd = loss_fd
        self.loss_lr = loss_lr
        self.loss_cmc = loss_cmc
        self.loss_fl = loss_fl
        self.loss_ar = loss_ar
        self.loss_kl = loss_kl

    def __iter__(self):
        return iter((
            self.total_loss,
            self.loss_fd,
            self.loss_lr,
            self.loss_cmc,
            self.loss_fl,
            self.loss_ar,
            self.loss_kl,
        ))


class MFVLRLoss(nn.Module):
    """Combined weighted loss module for MFVLR training."""

    def __init__(
        self,
        lambda_fd: float = 1.0,
        lambda_lr: float = 1.0,
        lambda_cmc: float = 1.0,
        lambda_fl: float = 1.0,
        lambda_ar: float = 1.0,
        lambda_kl: float = 1.0,
        kl_temperature: float = 0.5,
        cmc_initial_temperature: float = 0.07,
        cmc_trainable_temperature: bool = True,
        cmc_l2_normalize: bool = True,
        lr_ignore_index: int = -100,
        fd_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lambda_fd = lambda_fd
        self.lambda_lr = lambda_lr
        self.lambda_cmc = lambda_cmc
        self.lambda_fl = lambda_fl
        self.lambda_ar = lambda_ar
        self.lambda_kl = lambda_kl

        # Sub-loss modules
        self.fd_loss = ForgeryDetectionLoss(weight=fd_class_weights)
        self.lr_loss = LanguageReconstructionLoss(ignore_index=lr_ignore_index)
        self.ar_loss = AppearanceReconstructionLoss()
        self.fl_loss = ForgeryLocalizationLoss()
        self.kl_loss = KLSemanticAlignmentLoss(temperature=kl_temperature)
        self.cmc_loss = CrossModalContrastiveLoss(
            initial_temperature=cmc_initial_temperature,
            trainable_temperature=cmc_trainable_temperature,
            l2_normalize=cmc_l2_normalize,
        )

    def forward(
        self,
        y_pre: torch.Tensor,
        y_target: torch.Tensor,
        t_pre: torch.Tensor,
        target_token_ids: torch.Tensor,
        i_v: torch.Tensor,
        t_l: torch.Tensor,
        m_pre: torch.Tensor,
        target_mask: torch.Tensor,
        i_pre: torch.Tensor,
        image: torch.Tensor,
        t_lpre: torch.Tensor,
    ) -> TotalLossOutput:
        """Compute all six losses and weighted total loss (Eq. 27).

        Args:
            y_pre: Detection logits [B, 2]
            y_target: Detection target class indices [B]
            t_pre: Vocabulary logits [B, 308, 49408]
            target_token_ids: Ground-truth token IDs [B, 308]
            i_v: Global visual feature [B, 512]
            t_l: Global language feature [B, 512]
            m_pre: Predicted mask logits [B, 2, 224, 224]
            target_mask: Ground-truth binary mask [B, 224, 224]
            i_pre: Reconstructed appearance image [B, 3, 224, 224]
            image: Original input image [B, 3, 224, 224]
            t_lpre: Adapter predicted language feature [B, 512]

        Returns:
            TotalLossOutput containing total_loss and all individual loss terms.
        """
        # 1. Forgery Detection Loss
        loss_fd = self.fd_loss(y_pre, y_target)

        # 2. Language Reconstruction Loss
        loss_lr = self.lr_loss(t_pre, target_token_ids)

        # 3. Cross-Modal Contrastive Loss
        loss_cmc = self.cmc_loss(i_v, t_l)

        # 4. Forgery Localization Loss
        loss_fl = self.fl_loss(m_pre, target_mask)

        # 5. Appearance Reconstruction Loss
        loss_ar = self.ar_loss(i_pre, image)

        # 6. KL Semantic Alignment Loss
        loss_kl = self.kl_loss(t_l, t_lpre)

        # Total weighted loss (Eq. 27)
        total_loss = (
            self.lambda_fd * loss_fd
            + self.lambda_lr * loss_lr
            + self.lambda_cmc * loss_cmc
            + self.lambda_fl * loss_fl
            + self.lambda_ar * loss_ar
            + self.lambda_kl * loss_kl
        )

        return TotalLossOutput(
            total_loss=total_loss,
            loss_fd=loss_fd,
            loss_lr=loss_lr,
            loss_cmc=loss_cmc,
            loss_fl=loss_fl,
            loss_ar=loss_ar,
            loss_kl=loss_kl,
        )
