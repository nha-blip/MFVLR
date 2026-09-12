"""Random seed setting for reproducibility."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution.

    # ASSUMPTION_FROM_PAPER_GAP:
    # Paper does not specify seed value or determinism settings.
    # Reproduction defaults seed to 42 and sets PyTorch deterministic flags.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
