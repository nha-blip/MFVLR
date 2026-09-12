"""MFVLR model package.

Module implementation schedule:
- Phase 3: MVE (Image Encoder, Residual Encoder) [COMPLETE]
- Phase 4: VD (Appearance Decoder, Mask Decoder) [COMPLETE]
- Phase 5: VIM [COMPLETE]
- Phase 6: Language Encoder [COMPLETE]
- Phase 7: Language Decoder [COMPLETE]
- Phase 8: Losses & Adapter & FLT [COMPLETE]
- Phase 9: End-to-End MFVLR Model [COMPLETE]
- Phase 10: Training, Evaluation & Image-Only Inference [NEXT]
"""

from models.vision import (
    UNetEncoder,
    ImageTransformer,
    ImageEncoder,
    ResidualEncoder,
    MultiDomainVisionEncoder,
    UNetDecoderTrunk,
    AppearanceDecoder,
    AppearanceHead,
    MaskDecoder,
    MaskHead,
    VisionDecoder,
)
from models.language import (
    LanguageEmbeddings,
    VisionInjectionModule,
    VIM,
    LanguageEncoderBlock,
    LanguageEncoder,
    LanguageEncoderOutput,
    LanguageDecoderBlock,
    LanguageDecoder,
    LanguageDecoderOutput,
    FineGrainedLanguageTransformer,
    FLT,
    FLTOutput,
)
from models.heads import (
    Adapter,
    DetectionHead,
)
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
from models.mfvlr import (
    MFVLR,
    MFVLROutput,
    MFVLRInferenceOutput,
)

__all__ = [
    "UNetEncoder",
    "ImageTransformer",
    "ImageEncoder",
    "ResidualEncoder",
    "MultiDomainVisionEncoder",
    "UNetDecoderTrunk",
    "AppearanceDecoder",
    "AppearanceHead",
    "MaskDecoder",
    "MaskHead",
    "VisionDecoder",
    "LanguageEmbeddings",
    "VisionInjectionModule",
    "VIM",
    "LanguageEncoderBlock",
    "LanguageEncoder",
    "LanguageEncoderOutput",
    "LanguageDecoderBlock",
    "LanguageDecoder",
    "LanguageDecoderOutput",
    "FineGrainedLanguageTransformer",
    "FLT",
    "FLTOutput",
    "Adapter",
    "DetectionHead",
    "ForgeryDetectionLoss",
    "LanguageReconstructionLoss",
    "AppearanceReconstructionLoss",
    "ForgeryLocalizationLoss",
    "KLSemanticAlignmentLoss",
    "CrossModalContrastiveLoss",
    "MFVLRLoss",
    "TotalLossOutput",
    "MFVLR",
    "MFVLROutput",
    "MFVLRInferenceOutput",
]

