"""Image transformations for MFVLR.

PAPER_SPECIFIED:
- Image resolution: 3x224x224

ASSUMPTION_FROM_PAPER_GAP:
- Conversion to PyTorch FloatTensor in range [0, 1].
- Preprocessing and normalization details.
"""

from typing import Callable, Optional
import torchvision.transforms as T
from PIL import Image


def get_transforms(
    image_size: int = 224,
    is_train: bool = True,
    scale_to_unit_interval: bool = True,
) -> Callable[[Image.Image], Image.Image]:
    """Build standard image preprocessing transforms.

    # ASSUMPTION_FROM_PAPER_GAP:
    # Minimal transform pipeline: Resize to (224, 224) + ToTensor (yielding [0, 1]).
    # Augmentations disabled by default to stay faithful to paper baseline.
    """
    transform_list = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ]

    return T.Compose(transform_list)
