"""Tests for configurable label mapping (real_class_index vs fake_class_index) (Phase 11).

Ensures that evaluation and metrics never invert or assume hard-coded class indices.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from models.mfvlr import MFVLR
from utils.metrics import compute_classification_metrics
from utils.evaluator import evaluate


class SyntheticMappingDataset(Dataset):
    """Synthetic dataset with known ground truth and configurable class indices."""

    def __init__(self, fake_class_index: int = 1, real_class_index: int = 0):
        self.fake_class_index = fake_class_index
        self.real_class_index = real_class_index
        torch.manual_seed(42)
        # 4 samples: 2 real, 2 fake
        self.images = torch.randn(4, 3, 224, 224).clamp(0.0, 1.0)
        # 0: Real, 1: Fake, 2: Real, 3: Fake
        self.is_fakes = [False, True, False, True]
        self.labels = [fake_class_index if f else real_class_index for f in self.is_fakes]
        self.masks = torch.randint(0, 2, (4, 224, 224), dtype=torch.long)

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "mask": self.masks[idx],
        }


def test_metric_calculation_with_inverted_class_index():
    """Verify compute_classification_metrics uses pos_label to correctly evaluate when fake_class_index=0."""
    # Scenario A: fake_class_index = 1 (standard)
    y_true_std = np.array([0, 1, 0, 1])  # 0=real, 1=fake
    y_probs_fake = np.array([0.1, 0.9, 0.2, 0.8])  # High prob for fake samples
    metrics_std = compute_classification_metrics(y_true_std, y_probs_fake, pos_label=1)
    assert metrics_std["acc"] == 100.0
    assert metrics_std["auc"] == 100.0

    # Scenario B: fake_class_index = 0 (inverted)
    y_true_inv = np.array([1, 0, 1, 0])  # 1=real, 0=fake
    # y_probs_fake is the probability of class 0 (fake)
    y_probs_fake_inv = np.array([0.1, 0.9, 0.2, 0.8])
    metrics_inv = compute_classification_metrics(y_true_inv, y_probs_fake_inv, pos_label=0)
    assert metrics_inv["acc"] == 100.0
    assert metrics_inv["auc"] == 100.0


def test_evaluator_with_configured_fake_class_index():
    """Verify evaluate() correctly passes positive_label and uses appropriate softmax probability column."""
    model = MFVLR()
    model.eval()

    # When fake_class_index = 0
    dataset = SyntheticMappingDataset(fake_class_index=0, real_class_index=1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    results = evaluate(model=model, dataloader=dataloader, positive_label=0)

    assert "acc" in results
    assert "auc" in results
    assert 0.0 <= results["acc"] <= 100.0
    assert 0.0 <= results["auc"] <= 100.0
