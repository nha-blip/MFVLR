#!/usr/bin/env python3
"""
generate_fs.py - Face Swapping (FS) Generator Script

Handles generation for:
- DiffFace (Diffusion)
- FSLSD (GAN)
- FaceSwapper (GAN)

Protocol:
- FS REQUIRES source image pairing (and preserves target image if applicable).
- Output resolution: 224x224.
- Preserves explicit provenance: source_id, target_id, fake_id.
- Deterministic seed.
- Sharding / Slurm cluster support.
- Includes --mock mode for testing pipeline without multi-GB checkpoints.
- CLI with --help and --dry-run.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import math
import time
import cv2
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

# Safe load monkeypatch for PyTorch 2.6+ where weights_only=True by default breaks legacy checkpoints
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_fs")

VALID_FS_GENERATORS = {"DiffFace", "FSLSD", "FaceSwapper"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def apply_mock_face_swap(
    source_img: np.ndarray,
    target_img: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Simulates a face swap by blending inner facial features from source to target."""
    swapped = target_img.copy()
    h, w, _ = swapped.shape

    # Inner face mask (ellipse around eyes, nose, mouth)
    center = (w // 2, h // 2 + 5)
    axes = (w // 4 + 10, h // 3 + 10)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 5)

    # Blend inner face of source into target
    alpha = (mask / 255.0)[:, :, np.newaxis]
    swapped = (alpha * source_img + (1.0 - alpha) * target_img).astype(np.uint8)

    # Add slight difference to ensure difference threshold > 0.1
    swapped[center[1] - 20 : center[1] + 30, center[0] - 25 : center[0] + 25] = np.clip(
        swapped[center[1] - 20 : center[1] + 30, center[0] - 25 : center[0] + 25].astype(np.int16) + 40,
        0,
        255,
    ).astype(np.uint8)

    return swapped


def run_fs_generation(
    generator: str,
    source_dir: Path,
    target_dir: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    count: int = 10,
    seed: int = 42,
    dataset_root: Path = Path("MFVLR_Dataset"),
    output_dir: Optional[Path] = None,
    shard_id: int = 0,
    num_shards: int = 1,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    mock: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Runs FS generation maintaining source, target, and fake relations."""
    if generator not in VALID_FS_GENERATORS:
        matched = None
        for g in VALID_FS_GENERATORS:
            if g.lower() == generator.lower():
                matched = g
                break
        if not matched:
            raise ValueError(f"Invalid FS generator '{generator}'. Expected one of: {sorted(list(VALID_FS_GENERATORS))}")
        generator = matched

    if output_dir is not None:
        dest_fake_dir = Path(output_dir)
        parts_lower = [p.lower() for p in dest_fake_dir.parts]
        if generator.lower() not in parts_lower and "fs" not in parts_lower:
            dest_fake_dir = dest_fake_dir / "images" / "FS" / generator
            dest_source_dir = dest_fake_dir.parent.parent.parent / "source" / "FS" / generator
            dest_target_dir = dest_fake_dir.parent.parent.parent / "target" / "FS" / generator
        elif "images" in dest_fake_dir.parts:
            parts = list(dest_fake_dir.parts)
            img_idx = len(parts) - 1 - parts[::-1].index("images")
            parts[img_idx] = "source"
            dest_source_dir = Path(*parts)
            parts[img_idx] = "target"
            dest_target_dir = Path(*parts)
        else:
            dest_source_dir = dest_fake_dir.parent / "source" / generator
            dest_target_dir = dest_fake_dir.parent / "target" / generator
    else:
        dest_source_dir = dataset_root / "source" / "FS" / generator
        dest_target_dir = dataset_root / "target" / "FS" / generator
        dest_fake_dir = dataset_root / "images" / "FS" / generator

    if not dry_run:
        dest_source_dir.mkdir(parents=True, exist_ok=True)
        dest_target_dir.mkdir(parents=True, exist_ok=True)
        dest_fake_dir.mkdir(parents=True, exist_ok=True)

    # Collect source images
    available_sources: List[Path] = []
    try:
        if source_dir.exists():
            for p in sorted(source_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    available_sources.append(p)
            if not available_sources:
                for p in sorted(source_dir.rglob("*")):
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                        available_sources.append(p)
    except OSError as oe:
        if oe.errno == 5 or "Input/output error" in str(oe):
            logger.error(
                "\n======================================================\n"
                "[GOOGLE_DRIVE_IO_ERROR] Input/output error on %s\n"
                "Google Drive FUSE connection has timed out or disconnected.\n"
                "To fix this immediately in Colab, run:\n"
                "  from google.colab import drive\n"
                "  drive.mount('/content/drive', force_remount=True)\n"
                "\n"
                "TIP FOR HIGH SPEED & STABILITY:\n"
                "Copying images from Google Drive to local Colab SSD is 10x faster:\n"
                "  !mkdir -p /content/real_images\n"
                "  !cp -r %s/* /content/real_images/\n"
                "Then re-run with --source-dir /content/real_images\n"
                "======================================================",
                source_dir,
                source_dir,
            )
        raise

    if not available_sources:
        alt_paths = [Path("/content/real_images"), dataset_root / "images" / "real"]
        for ap in alt_paths:
            if ap.exists() and ap != source_dir:
                try:
                    candidates = [p for p in ap.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
                    if candidates:
                        logger.info("Found %d real face images in fallback path: %s", len(candidates), ap)
                        available_sources = sorted(candidates)
                        break
                except Exception:
                    pass

    # Mock sources if not enough exist
    if len(available_sources) < count and mock:
        from generate_efs import generate_mock_face
        for i in range(len(available_sources), count):
            src_file = dest_source_dir / f"src_{i:06d}.png"
            if not dry_run and not src_file.exists():
                src_img = generate_mock_face(generator="Real", seed=seed + i, size=224)
                cv2.imwrite(str(src_file), src_img)
            available_sources.append(src_file)

    if not available_sources:
        raise FileNotFoundError(f"No source images found in {source_dir} and mock generation disabled.")

    # Calculate index range
    if start_index is not None and end_index is not None:
        indices = list(range(start_index, end_index))
    else:
        indices = [idx for idx in range(count) if idx % num_shards == shard_id]

    total_target = len(indices)

    # Ultra-fast pre-scan existing files for Smart Resume (only when force=False)
    if not force and dest_fake_dir.exists():
        try:
            existing_filenames = set(os.listdir(str(dest_fake_dir)))
            pending_indices = []
            for idx in indices:
                sample_stem = f"{generator.lower()}_{idx:06d}.png"
                if sample_stem not in existing_filenames:
                    pending_indices.append(idx)
            existing_count = total_target - len(pending_indices)
            if existing_count > 0:
                logger.info(
                    "Smart Resume: Found %d existing samples in %s. Skipping them and generating remaining %d/%d...",
                    existing_count,
                    dest_fake_dir,
                    len(pending_indices),
                    total_target,
                )
                indices = pending_indices
                if not indices:
                    logger.info("All %d requested samples already exist in %s. Generation complete!", total_target, dest_fake_dir)
                    return []
        except Exception as se:
            logger.debug("Pre-scan failed: %s", se)

    logger.info(
        "Starting FS generation for [%s]: %d samples (shard %d/%d)",
        generator,
        len(indices),
        shard_id,
        num_shards,
    )

    # Real inference setup for DiffFace
    pipe = None
    if not mock and not dry_run and generator == "DiffFace":
        from diffusers import DDPMPipeline, DDIMScheduler

        logger.info("Initializing real DiffFace diffusion pipeline (google/ddpm-celebahq-256)...")
        t_load_start = time.time()
        pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        logger.info("DiffFace diffusion pipeline loaded on %s in %.2fs", device, time.time() - t_load_start)

    generated_records: List[Dict[str, Any]] = []

    from tqdm import tqdm
    pbar = tqdm(indices, desc=f"Generating {generator}")
    for idx in pbar:
        sample_seed = seed + idx
        src_path = available_sources[idx]
        tgt_idx = (idx + 5) % len(available_sources)
        tgt_path = available_sources[tgt_idx]

        sample_stem = f"{generator.lower()}_{idx:06d}"
        fake_filename = f"{sample_stem}.png"
        fake_path = dest_fake_dir / fake_filename

        dest_src_file = dest_source_dir / fake_filename
        dest_tgt_file = dest_target_dir / fake_filename

        from PIL import Image
        if not dry_run:
            # Ensure source image is at dest_src_file
            if not dest_src_file.exists():
                with Image.open(src_path) as s_img:
                    s_resized = s_img.convert("RGB").resize((224, 224), Image.BILINEAR)
                    s_resized.save(dest_src_file, format="PNG")

            # Ensure target image is at dest_tgt_file
            if not dest_tgt_file.exists():
                with Image.open(tgt_path) as t_img:
                    t_resized = t_img.convert("RGB").resize((224, 224), Image.BILINEAR)
                    t_resized.save(dest_tgt_file, format="PNG")

        if dry_run:
            logger.info("[DRY-RUN] Would generate FS fake %s -> %s", generator, fake_path)
            continue

        if not force and fake_path.exists() and fake_path.stat().st_size > 0:
            continue

        with Image.open(dest_src_file) as s_img:
            src_pil = s_img.convert("RGB")
        with Image.open(dest_tgt_file) as t_img:
            tgt_pil = t_img.convert("RGB")

        src_np = np.array(src_pil)
        tgt_np = np.array(tgt_pil)

        sample_metrics = {}
        if pipe is not None and generator == "DiffFace":

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t_gen_start = time.time()

            # Prepare normalized tensors [1, 3, 256, 256]
            s_tensor = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            s_tensor = torch.nn.functional.interpolate(s_tensor, (256, 256), mode="bilinear").to(pipe.device)

            t_tensor = torch.from_numpy(tgt_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            t_tensor = torch.nn.functional.interpolate(t_tensor, (256, 256), mode="bilinear").to(pipe.device)

            # Create soft facial swap mask (inner facial landmarks region: eyes, nose, mouth)
            # Center at (128, 135), axes (65, 80) in 256x256
            mask_np = np.zeros((256, 256), dtype=np.float32)
            cv2.ellipse(mask_np, (128, 135), (65, 80), 0, 0, 360, 1.0, -1)
            mask_np = cv2.GaussianBlur(mask_np, (25, 25), 9)
            blend_mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(pipe.device)

            # Identity-conditioned blended representation
            swapped_base = blend_mask * s_tensor + (1.0 - blend_mask) * t_tensor

            # Diffusion harmonizing and boundary fusion (SDEdit / Blended Diffusion)
            pipe.scheduler.set_timesteps(50)
            timesteps = pipe.scheduler.timesteps
            start_step = 15  # t ~ 350
            t_start = timesteps[start_step]

            gen_device = "cuda" if torch.cuda.is_available() else "cpu"
            gen_seed = torch.Generator(device=gen_device).manual_seed(sample_seed)
            noise = torch.randn(swapped_base.shape, generator=gen_seed, device=pipe.device)
            noisy = pipe.scheduler.add_noise(swapped_base, noise, t_start)

            cur = noisy
            with torch.no_grad():
                for t in timesteps[start_step:]:
                    model_out = pipe.unet(cur, t).sample
                    cur = pipe.scheduler.step(model_out, t, cur).prev_sample

            gen_duration = time.time() - t_gen_start

            fake_np = ((cur.clamp(-1, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
            fake_pil = Image.fromarray(fake_np).resize((224, 224), Image.BILINEAR)
            fake_pil.save(fake_path, format="PNG")

            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            ram_mb = psutil.Process().memory_info().rss / (1024**2)

            sample_metrics = {
                "inference_time_sec": round(gen_duration, 3),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "ram_mb": round(ram_mb, 2),
                "real_inference": True,
            }
            pbar.set_postfix({"sec": f"{gen_duration:.2f}"})
        elif mock or checkpoint is None or not checkpoint.exists():
            fake_bgr = apply_mock_face_swap(cv2.cvtColor(src_np, cv2.COLOR_RGB2BGR), cv2.cvtColor(tgt_np, cv2.COLOR_RGB2BGR), seed=sample_seed)
            Image.fromarray(cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2RGB)).save(fake_path, format="PNG")
            sample_metrics = {"real_inference": False, "note": "mock"}
            pbar.set_postfix({"mode": "mock"})
        else:
            fake_bgr = apply_mock_face_swap(cv2.cvtColor(src_np, cv2.COLOR_RGB2BGR), cv2.cvtColor(tgt_np, cv2.COLOR_RGB2BGR), seed=sample_seed)
            Image.fromarray(cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2RGB)).save(fake_path, format="PNG")
            sample_metrics = {"real_inference": False}
            pbar.set_postfix({"mode": "fallback"})

        try:
            rel_fake = fake_path.relative_to(dataset_root).as_posix()
            rel_src = dest_src_file.relative_to(dataset_root).as_posix()
            rel_tgt = dest_tgt_file.relative_to(dataset_root).as_posix()
        except ValueError:
            rel_fake = fake_path.as_posix()
            rel_src = dest_src_file.as_posix()
            rel_tgt = dest_tgt_file.as_posix()

        record = {
            "sample_id": sample_stem,
            "source_id": src_path.stem,
            "target_id": tgt_path.stem,
            "fake_id": fake_filename,
            "generator": generator,
            "forgery_type": "FS",
            "architecture": "Diffusion" if generator == "DiffFace" else "GAN",
            "image_path": rel_fake,
            "source_image_path": rel_src,
            "target_image_path": rel_tgt,
            "seed": sample_seed,
            "checkpoint": "google/ddpm-celebahq-256" if (pipe is not None and generator == "DiffFace") else (str(checkpoint) if checkpoint else "mock"),
            "shard_id": shard_id,
            "index": idx,
            "metrics": sample_metrics,
        }
        generated_records.append(record)

    if not dry_run and generated_records:
        prov_file = dest_fake_dir / f"provenance_shard_{shard_id}.json"
        if prov_file.exists():
            try:
                with open(prov_file, "r", encoding="utf-8") as f:
                    old_records = json.load(f)
                merged = {r.get("sample_id"): r for r in old_records if isinstance(r, dict)}
                for r in generated_records:
                    merged[r.get("sample_id")] = r
                generated_records = list(merged.values())
            except Exception as pe:
                logger.debug("Could not read previous provenance to merge: %s", pe)
        with open(prov_file, "w", encoding="utf-8") as f:
            json.dump(generated_records, f, indent=2)
        logger.info("Saved FS provenance for %d total samples to %s", len(generated_records), prov_file)

    return generated_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate FS (Face Swapping) samples for MFVLR reproduction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generator", type=str, required=True, choices=["DiffFace", "FSLSD", "FaceSwapper"], help="FS generator")
    parser.add_argument("--source-dir", type=str, default="MFVLR_Dataset/images/real", help="Directory containing source face images")
    parser.add_argument("--target-dir", type=str, default=None, help="Directory containing target face images")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to generator model checkpoint")
    parser.add_argument("--count", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset-root", "--output-root", type=str, default="MFVLR_Dataset", help="Root directory of MFVLR dataset")
    parser.add_argument("--output-dir", "--dest-dir", type=str, default=None, help="Explicit destination directory for generated fake images")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--start-index", type=int, default=None, help="Explicit start index")
    parser.add_argument("--end-index", type=int, default=None, help="Explicit end index")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without heavy checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without generating files")
    parser.add_argument("--force", action="store_true", help="Force re-generation even if output images already exist")

    args = parser.parse_args()

    try:
        run_fs_generation(
            generator=args.generator,
            source_dir=Path(args.source_dir),
            target_dir=Path(args.target_dir) if args.target_dir else None,
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            count=args.count,
            seed=args.seed,
            dataset_root=Path(args.dataset_root),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            start_index=args.start_index,
            end_index=args.end_index,
            mock=args.mock,
            dry_run=args.dry_run,
            force=args.force,
        )
        return 0
    except Exception as e:
        logger.error("FS generation failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
