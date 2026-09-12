"""Inference entry point for MFVLR single-image face forgery detection and localization.

Usage:
    python infer.py --config configs/mfvlr.yaml --checkpoint checkpoints/best_model.pt --image test_face.jpg
    python infer.py --config configs/mfvlr.yaml --checkpoint checkpoints/best_model.pt --image test_face.jpg --output-mask output_mask.png
"""

import argparse
import os
from typing import Any, Dict
from PIL import Image
import yaml
import numpy as np
import torch

from models.mfvlr import MFVLR, MFVLRInferenceOutput
from datasets.transforms import get_transforms
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image forgery detection and localization using MFVLR.")
    parser.add_argument("--config", type=str, default="configs/mfvlr.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--output-mask", type=str, default=None, help="Optional file path to save predicted mask image.")
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

    logger = setup_logger("MFVLR_Infer")

    # Validate image path
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image file not found at: {args.image}")

    # Resolve device
    device_str = args.device or config.get("training", {}).get("device")
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Label mappings
    real_class_idx = dataset_cfg.get("real_class_index", 0)
    fake_class_idx = dataset_cfg.get("fake_class_index", 1)

    # Build model and load weights
    model = MFVLR().to(device)
    model.eval()
    logger.info(f"Loading checkpoint weights from: {args.checkpoint}")
    load_checkpoint(filepath=args.checkpoint, model=model, device=device)

    # Load and preprocess input image [1, 3, 224, 224] in [0, 1]
    transform = get_transforms(image_size=224)
    raw_img = Image.open(args.image).convert("RGB")
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    # Execute paper-specified image-only inference (NO prompt, NO token IDs, NO tokenizer)
    with torch.no_grad():
        out: MFVLRInferenceOutput = model.forward_image_only(img_tensor, return_intermediates=True)

    # Process detection prediction
    logits = out.y_pre[0].cpu().numpy()
    probs = torch.softmax(out.y_pre, dim=1)[0].cpu().numpy()
    pred_idx = int(out.predict_class()[0].item())
    fake_prob = float(probs[fake_class_idx])

    pred_label = "Fake / Manipulated" if pred_idx == fake_class_idx else "Real / Pristine"

    # Process localization mask prediction
    pred_mask = out.predict_mask()[0].cpu().numpy().astype(np.uint8)  # [224, 224] in {0, 1}
    manipulated_pixels = int(pred_mask.sum())
    total_pixels = pred_mask.size
    manipulated_ratio = (manipulated_pixels / total_pixels) * 100.0

    logger.info("=" * 50)
    logger.info("MFVLR INFERENCE RESULTS:")
    logger.info(f"  Input Image:         {args.image}")
    logger.info(f"  Raw Logits (y_pre):  {logits}")
    logger.info(f"  Class Probabilities: Real: {probs[real_class_idx]:.4f}, Fake: {probs[fake_class_idx]:.4f}")
    logger.info(f"  Predicted Label:     {pred_label} (Class {pred_idx})")
    logger.info(f"  Fake Probability:    {fake_prob * 100:.2f}%")
    logger.info(f"  Manipulated Area:    {manipulated_pixels}/{total_pixels} pixels ({manipulated_ratio:.2f}%)")
    logger.info("=" * 50)

    # Optionally save mask image
    if args.output_mask:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_mask)) or ".", exist_ok=True)
        mask_img = Image.fromarray(pred_mask * 255)
        mask_img.save(args.output_mask)
        logger.info(f"Saved predicted localization mask to: {args.output_mask}")


if __name__ == "__main__":
    main()
