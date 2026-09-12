"""Evaluation entry point for MFVLR on face forgery detection and localization.

Usage:
    python evaluate.py --config configs/mfvlr.yaml --checkpoint checkpoints/best_model.pt
    python evaluate.py --config configs/mfvlr.yaml --checkpoint checkpoints/best_model.pt --manifest datasets/test_manifest.jsonl
"""

import argparse
import os
from typing import Any, Dict
import yaml
import torch
from torch.utils.data import DataLoader

from models.mfvlr import MFVLR
from datasets.dummy_dataset import DummyMFVLRDataset
from datasets.manifest_dataset import MFVLRDataset
from utils.checkpoint import load_checkpoint
from utils.evaluator import evaluate
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MFVLR model using image-only inference.")
    parser.add_argument("--config", type=str, default="configs/mfvlr.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to evaluation dataset manifest.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root directory for dataset paths.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default=None, help="Computation device ('cpu' or 'cuda').")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})

    logger = setup_logger("MFVLR_Eval")

    # Resolve device
    device_str = args.device or config.get("training", {}).get("device")
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    # Build evaluation dataset
    manifest_path = args.manifest or dataset_cfg.get("test_manifest") or dataset_cfg.get("val_manifest")
    dataset_root = args.dataset_root or dataset_cfg.get("root")
    real_class_idx = dataset_cfg.get("real_class_index", 0)
    fake_class_idx = dataset_cfg.get("fake_class_index", 1)

    if manifest_path and os.path.exists(manifest_path):
        eval_dataset = MFVLRDataset(
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            real_class_index=real_class_idx,
            fake_class_index=fake_class_idx,
        )
    else:
        logger.warning("No manifest provided or found. Using DummyMFVLRDataset for evaluation smoke test.")
        eval_dataset = DummyMFVLRDataset(num_samples=16, seed=123)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    logger.info(f"Evaluation dataset size: {len(eval_dataset)} samples.")

    # Build model and load checkpoint
    model = MFVLR().to(device)
    logger.info(f"Loading checkpoint weights from: {args.checkpoint}")
    load_checkpoint(filepath=args.checkpoint, model=model, device=device)

    # Execute paper-specified image-only evaluation
    logger.info("Executing image-only evaluation (no prompt, no tokenizer, no FLT)...")
    metrics = evaluate(
        model=model,
        dataloader=eval_loader,
        device=device,
        positive_label=fake_class_idx,
    )

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS:")
    logger.info(f"  Detection Accuracy (ACC): {metrics.get('acc', 0.0):.2f}%")
    logger.info(f"  Detection AUC Score:      {metrics.get('auc', 0.0):.2f}%")
    logger.info(f"  Localization mIoU:        {metrics.get('miou', 0.0):.2f}%")
    for k, v in metrics.items():
        if k not in ("acc", "auc", "miou"):
            logger.info(f"  {k}: {v:.2f}%")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
