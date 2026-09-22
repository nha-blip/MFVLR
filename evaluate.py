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

    # Resolve checkpoints to evaluate
    ckpt_paths: list[str] = []
    ckpt_input = args.checkpoint.strip()
    if os.path.isdir(ckpt_input):
        import glob
        found = glob.glob(os.path.join(ckpt_input, "*.pt"))
        import re
        def get_epoch(p):
            m = re.search(r"epoch_(\d+)", p)
            return int(m.group(1)) if m else 999999
        ckpt_paths = sorted(found, key=get_epoch)
    elif "*" in ckpt_input:
        import glob
        found = glob.glob(ckpt_input)
        import re
        def get_epoch(p):
            m = re.search(r"epoch_(\d+)", p)
            return int(m.group(1)) if m else 999999
        ckpt_paths = sorted(found, key=get_epoch)
    elif "," in ckpt_input:
        ckpt_paths = [c.strip() for c in ckpt_input.split(",") if c.strip()]
    else:
        ckpt_paths = [ckpt_input]

    if not ckpt_paths:
        logger.error(f"No checkpoint files found for: {args.checkpoint}")
        return

    # Build model once
    model = MFVLR().to(device)
    summary_results = []

    for idx, cp in enumerate(ckpt_paths, 1):
        if len(ckpt_paths) > 1:
            logger.info(f"\n[{idx}/{len(ckpt_paths)}] Evaluating checkpoint: {os.path.basename(cp)}")
        else:
            logger.info(f"Loading checkpoint weights from: {cp}")

        load_checkpoint(filepath=cp, model=model, device=device)

        logger.info("Executing image-only evaluation (no prompt, no tokenizer, no FLT)...")
        metrics = evaluate(
            model=model,
            dataloader=eval_loader,
            device=device,
            positive_label=fake_class_idx,
        )

        acc = metrics.get("acc", 0.0)
        auc = metrics.get("auc", 0.0)
        miou = metrics.get("miou", None)

        summary_results.append({
            "checkpoint": os.path.basename(cp),
            "acc": acc,
            "auc": auc,
            "miou": miou,
        })

        logger.info("=" * 50)
        logger.info(f"EVALUATION RESULTS for {os.path.basename(cp)}:")
        logger.info(f"  Detection Accuracy (ACC): {acc:.2f}%")
        logger.info(f"  Detection AUC Score:      {auc:.2f}%")
        if miou is not None:
            logger.info(f"  Localization mIoU:        {miou:.2f}%")
        for k, v in metrics.items():
            if k not in ("acc", "auc", "miou"):
                logger.info(f"  {k}: {v:.2f}%")
        logger.info("=" * 50)

    # Print summary table if evaluated multiple checkpoints
    if len(summary_results) > 1:
        print("\n" + "=" * 65)
        print("CHECKPOINT COMPARISON SUMMARY")
        print("=" * 65)
        print(f"{'Checkpoint':<30} | {'ACC (%)':<10} | {'AUC (%)':<10} | {'mIoU (%)':<10}")
        print("-" * 65)
        for r in summary_results:
            miou_str = f"{r['miou']:.2f}" if r["miou"] is not None else "N/A"
            print(f"{r['checkpoint']:<30} | {r['acc']:<10.2f} | {r['auc']:<10.2f} | {miou_str:<10}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
