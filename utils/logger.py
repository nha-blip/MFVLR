"""Logging setup for experiments."""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "MFVLR",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a standard logger.

    # ASSUMPTION_FROM_PAPER_GAP:
    # Logging formats and destinations are engineering choices.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
