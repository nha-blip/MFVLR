"""MFVLR Loss functions package."""

from models.losses.detection_loss import ForgeryDetectionLoss
from models.losses.language_reconstruction_loss import LanguageReconstructionLoss
from models.losses.appearance_reconstruction_loss import AppearanceReconstructionLoss
from models.losses.localization_loss import ForgeryLocalizationLoss
from models.losses.kl_loss import KLSemanticAlignmentLoss
from models.losses.cmc_loss import CrossModalContrastiveLoss
from models.losses.total_loss import MFVLRLoss, TotalLossOutput

__all__ = [
    "ForgeryDetectionLoss",
    "LanguageReconstructionLoss",
    "AppearanceReconstructionLoss",
    "ForgeryLocalizationLoss",
    "KLSemanticAlignmentLoss",
    "CrossModalContrastiveLoss",
    "MFVLRLoss",
    "TotalLossOutput",
]
