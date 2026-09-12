"""Prediction heads package (Adapter, DetectionHead)."""

from models.heads.adapter import Adapter
from models.heads.detection_head import DetectionHead

__all__ = [
    "Adapter",
    "DetectionHead",
]
