#!/usr/bin/env python3
"""
MFVLR / GenFace-Reproduced: Unified Resumable Generation Worker
Executes generation for a deterministic slice [start_idx, end_idx) of a given generator.
Designed for SLURM array jobs on Linux A100 clusters.
"""

import os
import sys
import time
import argparse
import hashlib
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image
import yaml

# Resolve roots from environment or defaults
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data")).resolve()
CHECKPOINT_ROOT = Path(os.environ.get("CHECKPOINT_ROOT", PROJECT_ROOT / "checkpoints")).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "output")).resolve()

METADATA_FIELDS = [
    "sample_id", "image_path", "source_path", "target_path", "mask_path",
    "label", "forgery_type", "architecture", "generator", "split",
    "L1", "L2", "L3", "L4", "original_dataset", "generator_repo",
    "generator_commit", "checkpoint", "checkpoint_sha256", "seed",
    "source_id", "target_id", "generation_config", "native_resolution",
    "sha256", "notes", "attribute", "attribute_strength"
]

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

def compute_difference_mask(fake_img: Image.Image, ref_img: Image.Image, threshold: float = 0.1) -> np.ndarray:
    fake_np = np.array(fake_img.convert("RGB")).astype(np.float32)
    ref_np = np.array(ref_img.convert("RGB")).astype(np.float32)
    diff = np.abs(fake_np - ref_np)
    gray = 0.299 * diff[:, :, 0] + 0.587 * diff[:, :, 1] + 0.114 * diff[:, :, 2]
    norm_gray = gray / 255.0
    mask = (norm_gray > threshold).astype(np.uint8) * 255
    return mask

def save_mask_lossless(mask_arr: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray(mask_arr, mode="L")
    tmp_path = out_path.with_suffix(".tmp.png")
    mask_img.save(tmp_path, format="PNG", compress_level=9)
    os.replace(tmp_path, out_path)

def save_image_atomic(img: Image.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.png")
    img.save(tmp_path, format="PNG")
    os.replace(tmp_path, out_path)

def is_sample_valid(img_path: Path, msk_path: Path) -> bool:
    if not img_path.exists() or not msk_path.exists():
        return False
    if img_path.stat().st_size == 0 or msk_path.stat().st_size == 0:
        return False
    try:
        with Image.open(img_path) as img:
            if img.size != (224, 224) or img.mode != "RGB":
                return False
        with Image.open(msk_path) as msk:
            if msk.size != (224, 224):
                return False
            m_np = np.array(msk)
            if not set(np.unique(m_np)).issubset({0, 255}):
                return False
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="MFVLR Production Generation Worker")
    parser.add_argument("--generator", type=str, required=True, help="Generator name (e.g. DDPM, StyleGAN3)")
    parser.add_argument("--start-index", type=int, required=True, help="Inclusive start sample index")
    parser.add_argument("--end-index", type=int, required=True, help="Exclusive end sample index")
    parser.add_argument("--config", type=str, required=True, help="Path to generator YAML config")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT), help="Output directory root")
    parser.add_argument("--checkpoint-root", type=str, default=str(CHECKPOINT_ROOT), help="Checkpoint directory root")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT), help="Pristine datasets directory root")
    parser.add_argument("--task-id", type=str, default="0", help="SLURM array task ID")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip already verified samples")
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    ckpt_root = Path(args.checkpoint_root).resolve()
    data_root = Path(args.data_root).resolve()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gen_name = cfg["generator"]
    cat = cfg["category"]
    arch = cfg["architecture"]
    prompts = cfg["prompts"]

    img_dir = out_root / "images" / cat / gen_name
    msk_dir = out_root / "masks" / cat / gen_name
    src_dir = out_root / "source" / cat / gen_name
    tar_dir = out_root / "target" / cat / gen_name
    meta_dir = out_root / "metadata" / gen_name

    for d in [img_dir, msk_dir, src_dir, tar_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    shard_path = meta_dir / f"shard_{args.task_id}.csv"
    print(f"=== Starting Task {args.task_id} for {gen_name} ===")
    print(f"Index range: [{args.start_index}, {args.end_index})")
    print(f"Shard output: {shard_path}")

    # Load shard if exists to resume records
    existing_records = {}
    if shard_path.exists():
        with open(shard_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_records[r["sample_id"]] = r

    f_shard = open(shard_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_shard, fieldnames=METADATA_FIELDS)
    writer.writeheader()

    # Re-write verified existing records
    for r in existing_records.values():
        writer.writerow(r)
    f_shard.flush()

    total_samples = args.end_index - args.start_index
    generated = 0
    skipped = 0

    print(f"Ready to process {total_samples} samples.")

    # Generator-specific execution will be hooked here
    # For now, record the execution boundary and ensure resume safety
    for idx in range(args.start_index, args.end_index):
        sample_id = f"{gen_name.lower()}_{idx:06d}"
        img_p = img_dir / f"{sample_id}.png"
        msk_p = msk_dir / f"{sample_id}.png"
        src_p = src_dir / f"{sample_id}.png"
        tar_p = tar_dir / f"{sample_id}.png"

        if args.resume and sample_id in existing_records:
            if is_sample_valid(img_p, msk_p):
                skipped += 1
                continue

        # In production worker, model generates image and mask
        # Placeholder for dispatch: actual generator models are called via modular dispatchers
        # when checkpoints and GPUs are mounted on the A100 server.

    f_shard.close()
    print(f"Task {args.task_id} finished: {generated} generated, {skipped} skipped.")

if __name__ == "__main__":
    main()
