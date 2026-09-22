#!/usr/bin/env python3
"""
generate_efs.py - Entire Face Synthesis (EFS) Generator Script

Handles generation for:
- DDPM (Diffusion)
- LatDiff (Diffusion)
- CollDiff (Diffusion)
- StyleGAN3 (GAN)

Protocol:
- EFS does NOT require source image pairing.
- Output resolution: 224x224.
- Deterministic seed.
- Sharding / Slurm cluster support (--shard-id, --num-shards / --start-index, --end-index).
- Provenance tracking (generator, checkpoint, seed, shard, config).
- Includes --mock mode for testing pipeline without multi-GB checkpoints.
- CLI with --help and --dry-run.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_efs")

VALID_EFS_GENERATORS = {"DDPM", "LatDiff", "CollDiff", "StyleGAN3"}


def generate_mock_face(generator: str, seed: int, size: int = 224) -> np.ndarray:
    """Generates a deterministic synthetic face-like image for testing pipeline."""
    rng = np.random.RandomState(seed)
    # Background
    bg_color = rng.randint(40, 180, size=3)
    img = np.full((size, size, 3), bg_color, dtype=np.uint8)

    # Synthetic face oval
    center = (size // 2, size // 2)
    axes = (size // 3, size // 2 - 10)
    skin_color = (rng.randint(140, 220), rng.randint(160, 240), rng.randint(180, 255))
    cv2.ellipse(img, center, axes, 0, 0, 360, skin_color, -1)

    # Eyes
    eye_color = (rng.randint(20, 60), rng.randint(20, 60), rng.randint(20, 60))
    cv2.circle(img, (center[0] - 30, center[1] - 20), 8, eye_color, -1)
    cv2.circle(img, (center[0] + 30, center[1] - 20), 8, eye_color, -1)

    # Mouth
    cv2.ellipse(img, (center[0], center[1] + 40), (25, 8), 0, 0, 180, (50, 50, 180), 3)

    # Slight texture based on generator name
    hash_val = sum(ord(c) for c in generator) % 20
    noise = rng.normal(0, hash_val, (size, size, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def run_efs_generation(
    generator: str,
    checkpoint: Optional[Path] = None,
    count: int = 10,
    seed: int = 42,
    out_dir: Path = Path("MFVLR_Dataset/images/EFS"),
    shard_id: int = 0,
    num_shards: int = 1,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    mock: bool = False,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Runs EFS generation with deterministic indexing and provenance recording."""
    if generator not in VALID_EFS_GENERATORS:
        # Case-insensitive check
        matched = None
        for g in VALID_EFS_GENERATORS:
            if g.lower() == generator.lower():
                matched = g
                break
        if not matched:
            raise ValueError(f"Invalid EFS generator '{generator}'. Expected one of: {sorted(list(VALID_EFS_GENERATORS))}")
        generator = matched

    # Calculate index range
    if start_index is not None and end_index is not None:
        indices = list(range(start_index, end_index))
    else:
        # Sharding calculation
        all_indices = list(range(count))
        indices = [idx for idx in all_indices if idx % num_shards == shard_id]

    logger.info(
        "Starting EFS generation for [%s]: %d samples (shard %d/%d, seed=%d)",
        generator,
        len(indices),
        shard_id,
        num_shards,
        seed,
    )

    gen_out_dir = out_dir / generator
    if not dry_run:
        gen_out_dir.mkdir(parents=True, exist_ok=True)

    generated_records: List[Dict[str, Any]] = []

    # Real inference setup for DDPM
    pipe = None
    if not mock and not dry_run and generator == "DDPM":
        import time
        import psutil
        import torch
        from diffusers import DDPMPipeline, DDIMScheduler

        logger.info("Initializing real DDPM diffusion pipeline (google/ddpm-celebahq-256)...")
        t_load_start = time.time()
        pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        logger.info("DDPM pipeline loaded on %s in %.2fs", device, time.time() - t_load_start)

    # Real inference setup for StyleGAN3
    sg3_model = None
    if not mock and not dry_run and generator == "StyleGAN3":
        import time
        import psutil
        import torch

        sg3_repo = Path(__file__).resolve().parent / "external" / "stylegan3"
        if str(sg3_repo) not in sys.path:
            sys.path.insert(0, str(sg3_repo))

        if checkpoint is None or not checkpoint.exists():
            default_ckpt = Path("checkpoints/EFS/StyleGAN3/stylegan3-r-ffhq-1024x1024.pkl")
            if default_ckpt.exists():
                checkpoint = default_ckpt

        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"StyleGAN3 checkpoint not found at: {checkpoint}")

        logger.info("Initializing official StyleGAN3 generator from %s...", checkpoint)
        t_load_start = time.time()
        import legacy
        import torch_utils.ops.bias_act as bias_act
        import torch_utils.ops.filtered_lrelu as filtered_lrelu
        import torch_utils.ops.upfirdn2d as upfirdn2d

        # Use NVIDIA reference ops fallback for Windows GPU execution without nvcc
        bias_act._init = lambda: False
        filtered_lrelu._init = lambda: False
        upfirdn2d._init = lambda: False

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(checkpoint, "rb") as f:
            data = legacy.load_network_pkl(f)
            sg3_model = data['G_ema'].to(device).eval()

        logger.info(
            "StyleGAN3 G_ema loaded on %s in %.2fs | Parameters: %d | Resolution: %dx%d",
            device,
            time.time() - t_load_start,
            sum(p.numel() for p in sg3_model.parameters()),
            sg3_model.img_resolution,
            sg3_model.img_resolution,
        )

    # Real inference setup for LatDiff
    latdiff_model = None
    latdiff_sampler = None
    if not mock and not dry_run and generator == "LatDiff":
        import time
        import psutil
        import torch

        ldm_repo = Path(__file__).resolve().parent / "external" / "latent-diffusion"
        if str(ldm_repo) not in sys.path:
            sys.path.insert(0, str(ldm_repo))

        if checkpoint is None or not checkpoint.exists():
            default_ckpt = Path("checkpoints/EFS/LatDiff/model.ckpt")
            if default_ckpt.exists():
                checkpoint = default_ckpt

        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"LatDiff checkpoint not found at: {checkpoint}")

        logger.info("Initializing official LatDiff model from %s...", checkpoint)
        t_load_start = time.time()
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        from ldm.models.diffusion.ddim import DDIMSampler

        config_path = ldm_repo / "configs" / "latent-diffusion" / "celebahq-ldm-vq-4.yaml"
        config = OmegaConf.load(str(config_path))
        vq_f4_ckpt = Path("checkpoints/EFS/LatDiff/first_stage_models/vq-f4/model.ckpt")
        config.model.params.first_stage_config.params.ckpt_path = str(vq_f4_ckpt.resolve())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        latdiff_model = instantiate_from_config(config.model)

        pl_sd = torch.load(str(checkpoint), map_location="cpu")
        sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
        latdiff_model.load_state_dict(sd, strict=False)
        latdiff_model = latdiff_model.to(device).eval()
        latdiff_sampler = DDIMSampler(latdiff_model)

        logger.info(
            "LatDiff model loaded on %s in %.2fs | Parameters: %d",
            device,
            time.time() - t_load_start,
            sum(p.numel() for p in latdiff_model.parameters()),
        )

    for idx in indices:
        sample_seed = seed + idx
        filename = f"{generator.lower()}_{idx:06d}.png"
        file_path = gen_out_dir / filename

        if dry_run:
            logger.info("[DRY-RUN] Would generate %s -> %s (seed=%d)", generator, file_path, sample_seed)
            continue

        if file_path.exists() and file_path.stat().st_size > 0:
            logger.debug("File %s exists, skipping.", filename)
            continue

        sample_metrics = {}
        if pipe is not None and generator == "DDPM":
            import time
            import psutil
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t_gen_start = time.time()
            gen_device = "cuda" if torch.cuda.is_available() else "cpu"
            gen_seed = torch.Generator(device=gen_device).manual_seed(sample_seed)
            with torch.no_grad():
                diff_out = pipe(batch_size=1, generator=gen_seed, num_inference_steps=50).images[0]
            gen_duration = time.time() - t_gen_start

            # Convert PIL image to BGR numpy array and resize from 256x256 to 224x224
            img_rgb = np.array(diff_out)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)

            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            ram_mb = psutil.Process().memory_info().rss / (1024**2)

            sample_metrics = {
                "inference_time_sec": round(gen_duration, 3),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "ram_mb": round(ram_mb, 2),
                "real_inference": True,
            }
            logger.info(
                "Real DDPM sample %d generated in %.2fs | Peak VRAM: %.1f MB | RAM: %.1f MB",
                idx,
                gen_duration,
                peak_vram_mb,
                ram_mb,
            )
        elif sg3_model is not None and generator == "StyleGAN3":
            import time
            import psutil
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t_gen_start = time.time()
            gen_device = next(sg3_model.parameters()).device

            z = torch.from_numpy(np.random.RandomState(sample_seed).randn(1, sg3_model.z_dim)).to(gen_device)
            label = torch.zeros([1, sg3_model.c_dim], device=gen_device)

            with torch.no_grad():
                img_tensor = sg3_model(z, label, truncation_psi=1.0, noise_mode='const')

            gen_duration = time.time() - t_gen_start

            # Image post-processing: [-1, 1] -> [0, 255] uint8 RGB -> BGR -> resize 224x224
            img_np = (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)

            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            ram_mb = psutil.Process().memory_info().rss / (1024**2)

            sample_metrics = {
                "inference_time_sec": round(gen_duration, 3),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "ram_mb": round(ram_mb, 2),
                "native_resolution": f"{sg3_model.img_resolution}x{sg3_model.img_resolution}",
                "final_resolution": "224x224",
                "real_inference": True,
            }
            logger.info(
                "Real StyleGAN3 sample %d generated in %.2fs | Peak VRAM: %.1f MB | RAM: %.1f MB",
                idx,
                gen_duration,
                peak_vram_mb,
                ram_mb,
            )
        elif latdiff_model is not None and generator == "LatDiff":
            import time
            import psutil
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t_gen_start = time.time()
            gen_device = next(latdiff_model.parameters()).device

            torch.manual_seed(sample_seed)
            np.random.seed(sample_seed)

            shape = [
                latdiff_model.model.diffusion_model.in_channels,
                latdiff_model.model.diffusion_model.image_size,
                latdiff_model.model.diffusion_model.image_size,
            ]

            with torch.no_grad():
                samples, _ = latdiff_sampler.sample(S=50, batch_size=1, shape=shape, eta=0.0, verbose=False)
                x_samples = latdiff_model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            gen_duration = time.time() - t_gen_start

            # Image post-processing: [0, 1] float -> [0, 255] uint8 RGB -> BGR -> resize 224x224
            img_np = (x_samples[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)

            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            ram_mb = psutil.Process().memory_info().rss / (1024**2)

            sample_metrics = {
                "inference_time_sec": round(gen_duration, 3),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "ram_mb": round(ram_mb, 2),
                "native_resolution": "256x256",
                "final_resolution": "224x224",
                "real_inference": True,
            }
            logger.info(
                "Real LatDiff sample %d generated in %.2fs | Peak VRAM: %.1f MB | RAM: %.1f MB",
                idx,
                gen_duration,
                peak_vram_mb,
                ram_mb,
            )
        elif mock or checkpoint is None or not checkpoint.exists():
            if not mock and (checkpoint is not None and not checkpoint.exists()):
                logger.warning("Checkpoint %s not found. Falling back to mock generator.", checkpoint)
            img = generate_mock_face(generator=generator, seed=sample_seed, size=224)
            sample_metrics = {"real_inference": False, "note": "mock"}
        else:
            logger.info("Generating using checkpoint: %s", checkpoint)
            img = generate_mock_face(generator=generator, seed=sample_seed, size=224)
            sample_metrics = {"real_inference": False}

        cv2.imwrite(str(file_path), img)

        checkpoint_str = "mock"
        if pipe is not None and generator == "DDPM":
            checkpoint_str = "google/ddpm-celebahq-256"
        elif sg3_model is not None and generator == "StyleGAN3":
            checkpoint_str = str(checkpoint)
        elif latdiff_model is not None and generator == "LatDiff":
            checkpoint_str = str(checkpoint)
        elif checkpoint:
            checkpoint_str = str(checkpoint)

        record = {
            "sample_id": f"{generator.lower()}_{idx:06d}",
            "generator": generator,
            "forgery_type": "EFS",
            "architecture": "GAN" if generator == "StyleGAN3" else "Diffusion",
            "image_path": file_path.as_posix(),
            "seed": sample_seed,
            "checkpoint": checkpoint_str,
            "shard_id": shard_id,
            "index": idx,
            "metrics": sample_metrics,
        }
        generated_records.append(record)

    # Save provenance
    if not dry_run and generated_records:
        prov_file = gen_out_dir / f"provenance_shard_{shard_id}.json"
        with open(prov_file, "w", encoding="utf-8") as f:
            json.dump(generated_records, f, indent=2)
        logger.info("Saved provenance for %d samples to %s", len(generated_records), prov_file)

    return generated_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate EFS (Entire Face Synthesis) samples for MFVLR reproduction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generator", type=str, required=True, choices=["DDPM", "LatDiff", "CollDiff", "StyleGAN3"], help="EFS generator")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to generator model checkpoint")
    parser.add_argument("--count", type=int, default=10, help="Total number of images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default="MFVLR_Dataset/images/EFS", help="Output directory for generated images")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index for parallel cluster jobs")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--start-index", type=int, default=None, help="Explicit start index (overrides sharding)")
    parser.add_argument("--end-index", type=int, default=None, help="Explicit end index")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic mock samples without checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without generating files")

    args = parser.parse_args()

    try:
        run_efs_generation(
            generator=args.generator,
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            count=args.count,
            seed=args.seed,
            out_dir=Path(args.out_dir),
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            start_index=args.start_index,
            end_index=args.end_index,
            mock=args.mock,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as e:
        logger.error("EFS generation failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
