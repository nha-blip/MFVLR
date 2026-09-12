"""Test module imports across the project."""

import pytest


def test_import_utils():
    """Verify utilities can be imported cleanly."""
    import utils
    from utils.seed import set_seed
    from utils.metrics import compute_classification_metrics, compute_localization_metrics
    from utils.checkpoint import save_checkpoint, load_checkpoint
    from utils.logger import setup_logger
    from utils.visualization import save_visualization, tensor_to_pil

    assert callable(set_seed)
    assert callable(compute_classification_metrics)
    assert callable(compute_localization_metrics)
    assert callable(save_checkpoint)
    assert callable(load_checkpoint)
    assert callable(setup_logger)


def test_import_datasets():
    """Verify dataset modules can be imported cleanly."""
    import datasets
    from datasets.dummy_dataset import DummyMFVLRDataset
    from datasets.genface import GenFaceDataset
    from datasets.prompt_generator import FineGrainedTextGenerator
    from datasets.mask_generator import generate_ground_truth_mask
    from datasets.transforms import get_transforms

    assert DummyMFVLRDataset is not None
    assert GenFaceDataset is not None
    assert FineGrainedTextGenerator is not None
    assert callable(generate_ground_truth_mask)
    assert callable(get_transforms)


def test_import_packages():
    """Verify models and losses packages are discoverable."""
    import models
    import models.vision
    import models.language
    import losses

    assert models is not None
    assert models.vision is not None
    assert models.language is not None
    assert losses is not None
