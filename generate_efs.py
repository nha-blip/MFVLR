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
import torch

# Safe load monkeypatch for PyTorch 2.6+ where weights_only=True by default breaks legacy checkpoints
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load


def load_drive_uploaded_indices(manifest_path: Optional[Path]) -> set[int]:
    """Load the set of sample indices already confirmed on Google Drive."""
    if manifest_path is None or not manifest_path.exists():
        return set()
    with manifest_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return {int(index) for index in data.get("uploaded_indices", [])}


def enqueue_drive_upload(image_path: Path, queue_dir: Optional[Path]) -> None:
    """Atomically publish a ready marker after a complete image is present."""
    if queue_dir is None:
        return
    queue_dir.mkdir(parents=True, exist_ok=True)
    marker = queue_dir / f"{image_path.stem}.ready"
    if marker.exists():
        return
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(image_path.name, encoding="utf-8")
    os.replace(temporary, marker)


def save_png_atomic(image: np.ndarray, image_path: Path) -> None:
    """Write a PNG to a temporary file and expose it only after completion."""
    temporary = image_path.with_name(f".{image_path.stem}.part.png")
    if not cv2.imwrite(str(temporary), image):
        temporary.unlink(missing_ok=True)
        raise OSError(f"Could not write generated image: {image_path}")
    os.replace(temporary, image_path)


def ensure_pytorch_lightning_compat() -> None:
    """Provides backward-compatibility shim for PyTorch Lightning 2.0+ (used by ldm in LatDiff and CollDiff)."""
    try:
        import pytorch_lightning.utilities.distributed
    except (ImportError, ModuleNotFoundError, AttributeError):
        import types
        try:
            from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_info
        except Exception:
            try:
                from lightning_utilities.core.rank_zero import rank_zero_only
                def rank_zero_info(*args, **kwargs):
                    pass
            except Exception:
                def rank_zero_only(fn):
                    return fn
                def rank_zero_info(*args, **kwargs):
                    pass

        dist_mod = types.ModuleType("pytorch_lightning.utilities.distributed")
        dist_mod.rank_zero_only = rank_zero_only
        dist_mod.rank_zero_info = rank_zero_info
        sys.modules["pytorch_lightning.utilities.distributed"] = dist_mod
        try:
            import pytorch_lightning.utilities
            pytorch_lightning.utilities.distributed = dist_mod
            if not hasattr(pytorch_lightning.utilities, "rank_zero_only"):
                pytorch_lightning.utilities.rank_zero_only = rank_zero_only
            if not hasattr(pytorch_lightning.utilities, "rank_zero_info"):
                pytorch_lightning.utilities.rank_zero_info = rank_zero_info
        except Exception:
            pass


ensure_pytorch_lightning_compat()


def ensure_colldiff_dependencies() -> None:
    """Ensures taming-transformers, CLIP, and kornia are available for LatDiff and CollDiff."""
    # 1. taming-transformers
    try:
        import taming
    except (ImportError, ModuleNotFoundError):
        external_dir = Path(__file__).resolve().parent / "external"
        taming_repo = external_dir / "taming-transformers"
        if not taming_repo.exists():
            logger.info("taming module not found. Auto-cloning CompVis/taming-transformers...")
            import subprocess
            try:
                external_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/CompVis/taming-transformers.git", str(taming_repo)],
                    check=True,
                )
            except Exception as e:
                logger.warning("Git clone taming-transformers failed: %s. Attempting pip install...", e)
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/CompVis/taming-transformers.git"], check=False)
                except Exception:
                    pass
        if taming_repo.exists() and str(taming_repo) not in sys.path:
            sys.path.insert(0, str(taming_repo))

    # 2. openai/CLIP
    try:
        import clip
    except (ImportError, ModuleNotFoundError):
        external_dir = Path(__file__).resolve().parent / "external"
        clip_repo = external_dir / "clip"
        if not clip_repo.exists():
            logger.info("clip module not found. Auto-cloning openai/CLIP...")
            import subprocess
            try:
                external_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/openai/CLIP.git", str(clip_repo)],
                    check=True,
                )
            except Exception as e:
                logger.warning("Git clone CLIP failed: %s. Attempting pip install...", e)
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"], check=False)
                except Exception:
                    pass
        if clip_repo.exists() and str(clip_repo) not in sys.path:
            sys.path.insert(0, str(clip_repo))

    # 3. kornia
    try:
        import kornia
    except (ImportError, ModuleNotFoundError):
        import subprocess
        logger.info("kornia not found. Auto-installing via pip...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "kornia"], check=False)
        except Exception as e:
            logger.warning("Failed to auto-install kornia: %s", e)

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
    batch_size: int = 4,
    fp16: bool = True,
    steps: int = 50,
    mock: bool = False,
    dry_run: bool = False,
    force: bool = False,
    drive_manifest: Optional[Path] = None,
    drive_queue_dir: Optional[Path] = None,
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

    parts_lower = [p.lower() for p in out_dir.parts]
    if generator.lower() not in parts_lower and "efs" not in parts_lower:
        gen_out_dir = out_dir / "images" / "EFS" / generator
    elif generator.lower() not in parts_lower:
        gen_out_dir = out_dir / generator
    else:
        gen_out_dir = out_dir

    if not dry_run:
        gen_out_dir.mkdir(parents=True, exist_ok=True)

    drive_uploaded_indices = load_drive_uploaded_indices(drive_manifest)

    # Ultra-fast pre-scan existing files for Smart Resume (only when force=False)
    if not force and gen_out_dir.exists():
        try:
            existing_filenames = set(os.listdir(str(gen_out_dir)))
            pending_indices = []
            for idx in indices:
                sample_stem = f"{generator.lower()}_{idx:06d}.png"
                if generator == "StyleGAN3" and idx in drive_uploaded_indices:
                    continue
                if sample_stem in existing_filenames:
                    if generator == "StyleGAN3" and not dry_run:
                        enqueue_drive_upload(gen_out_dir / sample_stem, drive_queue_dir)
                    continue
                else:
                    pending_indices.append(idx)
            existing_count = len(indices) - len(pending_indices)
            if existing_count > 0:
                logger.info(
                    "Smart Resume: Found %d existing samples in %s. Skipping them and generating remaining %d/%d...",
                    existing_count,
                    gen_out_dir,
                    len(pending_indices),
                    len(indices),
                )
                indices = pending_indices
                if not indices:
                    logger.info("All requested samples already exist in %s. Generation complete!", gen_out_dir)
                    return []
        except Exception as se:
            logger.debug("Pre-scan failed: %s", se)

    logger.info(
        "Starting EFS generation for [%s]: %d samples (shard %d/%d, seed=%d)",
        generator,
        len(indices),
        shard_id,
        num_shards,
        seed,
    )

    generated_records: List[Dict[str, Any]] = []

    # Real inference setup for DDPM
    pipe = None
    if not mock and not dry_run and generator == "DDPM":
        import time
        import psutil
        from diffusers import DDPMPipeline, DDIMScheduler

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        logger.info("Initializing real DDPM diffusion pipeline (google/ddpm-celebahq-256) in %s...", dtype)
        t_load_start = time.time()
        pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256", torch_dtype=dtype)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        logger.info("DDPM pipeline loaded on %s in %.2fs", device, time.time() - t_load_start)

    # Real inference setup for StyleGAN3
    sg3_model = None
    if not mock and not dry_run and generator == "StyleGAN3":
        import time
        import psutil
        sg3_repo = Path(__file__).resolve().parent / "external" / "stylegan3"
        if not (sg3_repo / "legacy.py").exists():
            logger.info("StyleGAN3 repository not found at %s. Auto-cloning from NVlabs/stylegan3...", sg3_repo)
            import subprocess
            sg3_repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "https://github.com/NVlabs/stylegan3.git", str(sg3_repo)],
                check=True
            )
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

        ldm_repo = Path(__file__).resolve().parent / "external" / "latent-diffusion"
        if not (ldm_repo / "ldm").exists():
            logger.info("Latent Diffusion repository not found at %s. Auto-cloning from CompVis/latent-diffusion...", ldm_repo)
            import subprocess
            ldm_repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "https://github.com/CompVis/latent-diffusion.git", str(ldm_repo)],
                check=True
            )
        if str(ldm_repo) not in sys.path:
            sys.path.insert(0, str(ldm_repo))

        if checkpoint is None or not checkpoint.exists():
            default_ckpt = Path("checkpoints/EFS/LatDiff/model.ckpt")
            if default_ckpt.exists():
                checkpoint = default_ckpt

        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"LatDiff checkpoint not found at: {checkpoint}")

        checkpoint = checkpoint.resolve()
        logger.info("Initializing official LatDiff model from %s...", checkpoint)
        t_load_start = time.time()

        # PyTorch Lightning 2.0+ compatibility shim for latent-diffusion
        ensure_pytorch_lightning_compat()
        ensure_colldiff_dependencies()

        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        from ldm.models.diffusion.ddim import DDIMSampler

        config_path = ldm_repo / "configs" / "latent-diffusion" / "celebahq-ldm-vq-4.yaml"
        config = OmegaConf.load(str(config_path))
        vq_f4_ckpt = Path("checkpoints/EFS/LatDiff/first_stage_models/vq-f4/model.ckpt")
        if not vq_f4_ckpt.exists():
            logger.info("First-stage VQ-f4 model not found at %s. Auto-downloading...", vq_f4_ckpt)
            import urllib.request
            import zipfile
            vq_f4_ckpt.parent.mkdir(parents=True, exist_ok=True)
            vq_zip = vq_f4_ckpt.parent / "vq-f4.zip"
            urllib.request.urlretrieve("https://ommer-lab.com/files/latent-diffusion/vq-f4.zip", str(vq_zip))
            with zipfile.ZipFile(vq_zip, "r") as zf:
                zf.extractall(vq_f4_ckpt.parent)
            logger.info("First-stage VQ-f4 model downloaded and extracted successfully.")
        config.model.params.first_stage_config.params.ckpt_path = str(vq_f4_ckpt.resolve())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        latdiff_model = instantiate_from_config(config.model)

        try:
            pl_sd = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        except TypeError:
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

    # Real inference setup for CollDiff
    colldiff_model = None
    if not mock and not dry_run and generator == "CollDiff":
        import time
        import psutil

        colldiff_repo = Path(__file__).resolve().parent / "external" / "colldiff"
        if not colldiff_repo.exists():
            logger.info("CollDiff repository not found at %s. Auto-cloning from ziqihuangg/Collaborative-Diffusion...", colldiff_repo)
            import subprocess
            colldiff_repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "https://github.com/ziqihuangg/Collaborative-Diffusion.git", str(colldiff_repo)],
                check=True
            )
        if str(colldiff_repo) not in sys.path:
            sys.path.insert(0, str(colldiff_repo))

        # Ensure colldiff_repo / pretrained has all checkpoints linked or copied
        pretrained_dir = colldiff_repo / "pretrained"
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        clean_target_names = [
            "256_codiff_mask_text.ckpt",
            "256_mask.ckpt",
            "256_text.ckpt",
            "256_vae.ckpt",
        ]

        # Clean up any accidental directory creation with .ckpt name
        for ctn in clean_target_names:
            p_dst = pretrained_dir / ctn
            if p_dst.exists() and p_dst.is_dir() and not p_dst.is_symlink():
                logger.warning("Directory found at %s. Removing invalid directory.", p_dst)
                shutil.rmtree(p_dst)

        candidate_ckpt_dirs = [
            Path("/content/drive/MyDrive"),
            Path("/content/drive/MyDrive/checkpoints"),
            Path("/content/drive/MyDrive/checkpoints/EFS/CollDiff"),
            Path("/content/drive/MyDrive/GenFace/checkpoints/EFS/CollDiff"),
            Path("/content/drive/MyDrive/GenFace"),
            Path("/content/MFVLR/checkpoints/EFS/CollDiff"),
            Path("checkpoints/EFS/CollDiff"),
        ]

        for cdir in candidate_ckpt_dirs:
            if cdir.exists():
                for ckpt_file in cdir.glob("*.ckpt"):
                    if not ckpt_file.is_file():
                        continue
                    # Normalize 'Copy of ...' or 'Bản sao của ...' to canonical filenames
                    canonical_name = ckpt_file.name
                    for ctn in clean_target_names:
                        if ctn in ckpt_file.name:
                            canonical_name = ctn
                            break
                    dst = pretrained_dir / canonical_name
                    if dst.exists() and dst.is_dir() and not dst.is_symlink():
                        shutil.rmtree(dst)
                    elif dst.is_symlink() or dst.exists():
                        try:
                            dst.unlink()
                        except Exception:
                            pass
                    try:
                        os.symlink(ckpt_file.resolve(), dst)
                        logger.info("Symlinked CollDiff checkpoint: %s -> %s", ckpt_file.name, dst)
                    except Exception:
                        try:
                            shutil.copyfile(str(ckpt_file), str(dst))
                            logger.info("Copied CollDiff checkpoint: %s -> %s", ckpt_file.name, dst)
                        except Exception as ce:
                            logger.debug("Could not link/copy %s to %s: %s", ckpt_file, dst, ce)

        # Reset checkpoint if a directory was mistakenly passed (e.g. output dir)
        if checkpoint is not None and not checkpoint.is_file():
            logger.warning("Passed checkpoint '%s' is a directory, not a .ckpt file. Auto-discovering valid checkpoint...", checkpoint)
            checkpoint = None

        if checkpoint is None or not checkpoint.is_file():
            candidates = [
                colldiff_repo / "pretrained" / "256_codiff_mask_text.ckpt",
                Path("/content/drive/MyDrive/Bản sao của 256_codiff_mask_text.ckpt"),
                Path("/content/drive/MyDrive/Copy of 256_codiff_mask_text.ckpt"),
                Path("/content/drive/MyDrive/256_codiff_mask_text.ckpt"),
                Path("/content/drive/MyDrive/checkpoints/EFS/CollDiff/256_codiff_mask_text.ckpt"),
                Path("/content/MFVLR/checkpoints/EFS/CollDiff/256_codiff_mask_text.ckpt"),
                Path("checkpoints/EFS/CollDiff/256_codiff_mask_text.ckpt"),
            ]
            for c in candidates:
                if c.is_file():
                    checkpoint = c
                    break

        if checkpoint is None or not checkpoint.is_file():
            if not mock:
                raise FileNotFoundError(
                    "CollDiff checkpoint '256_codiff_mask_text.ckpt' not found in checkpoints/EFS/CollDiff/ or Google Drive! "
                    "Please download all official CollDiff checkpoints by running:\n"
                    "  !python download_data.py --generator CollDiff"
                )
            logger.warning("CollDiff checkpoint not found. Falling back to mock unless checkpoint is supplied.")
        else:
            checkpoint = checkpoint.resolve()
            logger.info("Initializing official CollDiff model from %s...", checkpoint)
            t_load_start = time.time()
            ensure_pytorch_lightning_compat()
            ensure_colldiff_dependencies()
            from omegaconf import OmegaConf
            from ldm.util import instantiate_from_config

            config_path = colldiff_repo / "configs" / "256_codiff_mask_text.yaml"
            config = OmegaConf.load(str(config_path))

            # Resolve checkpoints to absolute filepaths
            seg_ckpt = (colldiff_repo / "pretrained" / "256_mask.ckpt").resolve()
            if not seg_ckpt.is_file():
                for alt_s in [Path("/content/drive/MyDrive/Bản sao của 256_mask.ckpt"), Path("/content/drive/MyDrive/256_mask.ckpt")]:
                    if alt_s.is_file(): seg_ckpt = alt_s.resolve(); break

            text_ckpt = (colldiff_repo / "pretrained" / "256_text.ckpt").resolve()
            if not text_ckpt.is_file():
                for alt_t in [Path("/content/drive/MyDrive/Bản sao của 256_text.ckpt"), Path("/content/drive/MyDrive/256_text.ckpt")]:
                    if alt_t.is_file(): text_ckpt = alt_t.resolve(); break

            config.model.params.seg_mask_ldm_config_path = str((colldiff_repo / "configs" / "256_mask.yaml").resolve())
            config.model.params.seg_mask_ldm_ckpt_path = str(seg_ckpt)
            config.model.params.text_ldm_config_path = str((colldiff_repo / "configs" / "256_text.yaml").resolve())
            config.model.params.text_ldm_ckpt_path = str(text_ckpt)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            old_cwd = os.getcwd()
            os.chdir(str(colldiff_repo))
            try:
                colldiff_model = instantiate_from_config(config.model)
                colldiff_model.init_from_ckpt(str(checkpoint))
                colldiff_model = colldiff_model.to(device).eval()
            finally:
                os.chdir(old_cwd)

            logger.info(
                "CollDiff model loaded on %s in %.2fs | Parameters: %d",
                device,
                time.time() - t_load_start,
                sum(p.numel() for p in colldiff_model.parameters()),
            )

            # Pre-cache sampler and conditioning to maximize inference speed across batches
            from ldm.models.diffusion.ddim_confidence import DDIMConfidenceSampler
            colldiff_sampler = DDIMConfidenceSampler(model=colldiff_model, return_confidence_map=False)
            colldiff_mask_path = colldiff_repo / "test_data" / "256_masks" / "29980.png"
            colldiff_input_text = "A photo of a face."
            if colldiff_mask_path.exists():
                from PIL import Image
                import torch.nn.functional as F
                with open(colldiff_mask_path, "rb") as f:
                    m_img = Image.open(f).resize((32, 32), Image.NEAREST)
                    flat = list(m_img.getdata())
                flat_t = torch.tensor(flat)
                colldiff_one_hot = F.one_hot(flat_t, num_classes=19).transpose(0, 1).unsqueeze(0).to(device)
            else:
                colldiff_one_hot = torch.zeros((1, 19, 1024), device=device)
            colldiff_cond_cache = {}

    # Filter pending indices that need generation (for resume support)
    pending_indices = []
    for idx in indices:
        filename = f"{generator.lower()}_{idx:06d}.png"
        file_path = gen_out_dir / filename
        if dry_run:
            logger.info("[DRY-RUN] Would generate %s -> %s (seed=%d)", generator, file_path, seed + idx)
            continue
        if generator == "StyleGAN3" and idx in drive_uploaded_indices:
            continue
        if not force and file_path.exists() and file_path.stat().st_size > 0:
            logger.debug("File %s exists, skipping.", filename)
            if generator == "StyleGAN3":
                enqueue_drive_upload(file_path, drive_queue_dir)
            continue
        pending_indices.append(idx)

    if dry_run:
        return []

    logger.info("Pending samples to generate for [%s]: %d / %d (batch_size=%d, fp16=%s)",
                generator, len(pending_indices), len(indices), batch_size, fp16)

    effective_bs = max(1, batch_size) if (pipe is not None or sg3_model is not None or latdiff_model is not None or colldiff_model is not None) else 1

    for b_start in range(0, len(pending_indices), effective_bs):
        batch_ids = pending_indices[b_start:b_start + effective_bs]
        cur_b = len(batch_ids)
        batch_seed = seed + batch_ids[0]

        if pipe is not None and generator == "DDPM":
            import time
            import psutil

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.backends.cudnn.benchmark = True
            t_gen_start = time.time()
            gen_device = "cuda" if torch.cuda.is_available() else "cpu"
            gen_seed = torch.Generator(device=gen_device).manual_seed(batch_seed)
            with torch.no_grad():
                diff_outs = pipe(batch_size=cur_b, generator=gen_seed, num_inference_steps=steps).images
            total_duration = time.time() - t_gen_start
            per_img_duration = round(total_duration / cur_b, 3)

            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            ram_mb = psutil.Process().memory_info().rss / (1024**2)

            for b_i, idx in enumerate(batch_ids):
                diff_out = diff_outs[b_i]
                img_rgb = np.array(diff_out)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)

                file_path = gen_out_dir / f"{generator.lower()}_{idx:06d}.png"
                cv2.imwrite(str(file_path), img)

                sample_metrics = {
                    "inference_time_sec": per_img_duration,
                    "peak_vram_mb": round(peak_vram_mb, 2),
                    "ram_mb": round(ram_mb, 2),
                    "batch_size": cur_b,
                    "real_inference": True,
                }
                record = {
                    "sample_id": f"{generator.lower()}_{idx:06d}",
                    "generator": generator,
                    "forgery_type": "EFS",
                    "architecture": "Diffusion",
                    "image_path": file_path.as_posix(),
                    "seed": seed + idx,
                    "checkpoint": "google/ddpm-celebahq-256",
                    "shard_id": shard_id,
                    "index": idx,
                    "metrics": sample_metrics,
                }
                generated_records.append(record)

            logger.info(
                "Real DDPM generated batch of %d (samples %d-%d) in %.2fs (%.2fs/img) | Peak VRAM: %.1f MB",
                cur_b,
                batch_ids[0],
                batch_ids[-1],
                total_duration,
                per_img_duration,
                peak_vram_mb,
            )

        elif sg3_model is not None and generator == "StyleGAN3":
            import time
            import psutil

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t_gen_start = time.time()
            gen_device = next(sg3_model.parameters()).device

            z = torch.from_numpy(np.random.RandomState(batch_seed).randn(cur_b, sg3_model.z_dim)).to(gen_device)
            label = torch.zeros([cur_b, sg3_model.c_dim], device=gen_device)

            # StyleGAN3 generates high-res 1024x1024 images. Sub-batch chunking of 2 prevents CUDA OOM on 15GB GPUs (T4).
            sub_b = min(cur_b, 2)
            out_list = []
            with torch.no_grad():
                for sub_start in range(0, cur_b, sub_b):
                    sub_end = min(sub_start + sub_b, cur_b)
                    try:
                        sub_out = sg3_model(z[sub_start:sub_end], label[sub_start:sub_end], truncation_psi=1.0, noise_mode='const')
                    except torch.OutOfMemoryError:
                        logger.warning("CUDA OOM at sub-batch %d; clearing cache and falling back to single-image generation", sub_b)
                        torch.cuda.empty_cache()
                        sub_out_single = []
                        for single_i in range(sub_start, sub_end):
                            sub_out_single.append(sg3_model(z[single_i:single_i+1], label[single_i:single_i+1], truncation_psi=1.0, noise_mode='const'))
                        sub_out = torch.cat(sub_out_single, dim=0)
                    out_list.append(sub_out)
            img_tensors = torch.cat(out_list, dim=0) if len(out_list) > 1 else out_list[0]

            total_duration = time.time() - t_gen_start
            per_img_duration = round(total_duration / cur_b, 3)

            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            ram_mb = psutil.Process().memory_info().rss / (1024**2)

            for b_i, idx in enumerate(batch_ids):
                img_np = (img_tensors[b_i:b_i+1].permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)

                file_path = gen_out_dir / f"{generator.lower()}_{idx:06d}.png"
                save_png_atomic(img, file_path)
                enqueue_drive_upload(file_path, drive_queue_dir)

                sample_metrics = {
                    "inference_time_sec": per_img_duration,
                    "peak_vram_mb": round(peak_vram_mb, 2),
                    "ram_mb": round(ram_mb, 2),
                    "batch_size": cur_b,
                    "native_resolution": f"{sg3_model.img_resolution}x{sg3_model.img_resolution}",
                    "final_resolution": "224x224",
                    "real_inference": True,
                }
                record = {
                    "sample_id": f"{generator.lower()}_{idx:06d}",
                    "generator": generator,
                    "forgery_type": "EFS",
                    "architecture": "GAN",
                    "image_path": file_path.as_posix(),
                    "seed": seed + idx,
                    "checkpoint": str(checkpoint),
                    "shard_id": shard_id,
                    "index": idx,
                    "metrics": sample_metrics,
                }
                generated_records.append(record)

            logger.info(
                "Real StyleGAN3 generated batch of %d (samples %d-%d) in %.2fs (%.3fs/img) | Peak VRAM: %.1f MB",
                cur_b,
                batch_ids[0],
                batch_ids[-1],
                total_duration,
                per_img_duration,
                peak_vram_mb,
            )

        elif latdiff_model is not None and generator == "LatDiff":
            import time
            import psutil

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t_gen_start = time.time()
            gen_device = next(latdiff_model.parameters()).device

            torch.manual_seed(batch_seed)
            np.random.seed(batch_seed)

            shape = [
                latdiff_model.model.diffusion_model.in_channels,
                latdiff_model.model.diffusion_model.image_size,
                latdiff_model.model.diffusion_model.image_size,
            ]

            with torch.no_grad():
                samples, _ = latdiff_sampler.sample(S=steps, batch_size=cur_b, shape=shape, eta=0.0, verbose=False)
                x_samples = latdiff_model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            total_duration = time.time() - t_gen_start
            per_img_duration = round(total_duration / cur_b, 3)
            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            ram_mb = psutil.Process().memory_info().rss / (1024**2)

            for b_i, idx in enumerate(batch_ids):
                sample_seed = seed + idx
                file_path = gen_out_dir / f"{generator.lower()}_{idx:06d}.png"
                img_np = (x_samples[b_i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(file_path), img)

                sample_metrics = {
                    "inference_time_sec": per_img_duration,
                    "peak_vram_mb": round(peak_vram_mb, 2),
                    "ram_mb": round(ram_mb, 2),
                    "batch_size": cur_b,
                    "native_resolution": "256x256",
                    "final_resolution": "224x224",
                    "real_inference": True,
                }
                record = {
                    "sample_id": f"{generator.lower()}_{idx:06d}",
                    "generator": generator,
                    "forgery_type": "EFS",
                    "architecture": "Diffusion",
                    "image_path": file_path.as_posix(),
                    "seed": sample_seed,
                    "checkpoint": str(checkpoint),
                    "shard_id": shard_id,
                    "index": idx,
                    "metrics": sample_metrics,
                }
                generated_records.append(record)

            logger.info(
                "Real LatDiff generated batch of %d (samples %d-%d) in %.2fs (%.3fs/img) | Peak VRAM: %.1f MB",
                cur_b,
                batch_ids[0],
                batch_ids[-1],
                total_duration,
                per_img_duration,
                peak_vram_mb,
            )

        elif colldiff_model is not None and generator == "CollDiff":
            import time
            import psutil

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.backends.cudnn.benchmark = True
            t_gen_start = time.time()
            gen_device = next(colldiff_model.parameters()).device

            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)

            # Use cached conditioning tensor across batches to eliminate repeated BERT & encoder overhead
            if cur_b not in colldiff_cond_cache:
                condition = {
                    "seg_mask": colldiff_one_hot.repeat(cur_b, 1, 1),
                    "text": [colldiff_input_text.lower()] * cur_b,
                }
                with torch.no_grad():
                    with colldiff_model.ema_scope("Plotting"):
                        colldiff_cond_cache[cur_b] = colldiff_model.get_learned_conditioning(condition)

            cond = colldiff_cond_cache[cur_b]

            with torch.no_grad():
                with colldiff_model.ema_scope("Plotting"):
                    z_0, _ = colldiff_sampler.sample(
                        S=steps,
                        batch_size=cur_b,
                        shape=(3, 64, 64),
                        conditioning=cond,
                        verbose=False,
                        eta=1.0,
                        log_every_t=steps + 1,
                    )
                x_samples = colldiff_model.decode_first_stage(z_0)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            total_duration = time.time() - t_gen_start
            per_img_duration = round(total_duration / cur_b, 3)
            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            ram_mb = psutil.Process().memory_info().rss / (1024**2)

            for b_i, idx in enumerate(batch_ids):
                sample_seed = seed + idx
                file_path = gen_out_dir / f"{generator.lower()}_{idx:06d}.png"
                img_np = (x_samples[b_i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(file_path), img)

                sample_metrics = {
                    "inference_time_sec": per_img_duration,
                    "peak_vram_mb": round(peak_vram_mb, 2),
                    "ram_mb": round(ram_mb, 2),
                    "batch_size": cur_b,
                    "native_resolution": "256x256",
                    "final_resolution": "224x224",
                    "real_inference": True,
                }
                record = {
                    "sample_id": f"{generator.lower()}_{idx:06d}",
                    "generator": generator,
                    "forgery_type": "EFS",
                    "architecture": "Diffusion",
                    "image_path": file_path.as_posix(),
                    "seed": sample_seed,
                    "checkpoint": str(checkpoint),
                    "shard_id": shard_id,
                    "index": idx,
                    "metrics": sample_metrics,
                }
                generated_records.append(record)

            logger.info(
                "Real CollDiff generated batch of %d (samples %d-%d) in %.2fs (%.3fs/img) | Peak VRAM: %.1f MB",
                cur_b,
                batch_ids[0],
                batch_ids[-1],
                total_duration,
                per_img_duration,
                peak_vram_mb,
            )

        else:
            # Mock or fallback
            for idx in batch_ids:
                sample_seed = seed + idx
                file_path = gen_out_dir / f"{generator.lower()}_{idx:06d}.png"
                img = generate_mock_face(generator=generator, seed=sample_seed, size=224)
                cv2.imwrite(str(file_path), img)
                record = {
                    "sample_id": f"{generator.lower()}_{idx:06d}",
                    "generator": generator,
                    "forgery_type": "EFS",
                    "architecture": "GAN" if generator == "StyleGAN3" else "Diffusion",
                    "image_path": file_path.as_posix(),
                    "seed": sample_seed,
                    "checkpoint": "mock",
                    "shard_id": shard_id,
                    "index": idx,
                    "metrics": {"real_inference": False},
                }
                generated_records.append(record)

    # Save provenance (merge with existing records if resuming)
    if not dry_run and generated_records:
        prov_file = gen_out_dir / f"provenance_shard_{shard_id}.json"
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
        logger.info("Saved provenance for %d total samples to %s", len(generated_records), prov_file)

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
    parser.add_argument("--out-dir", "--output-dir", "--dest-dir", dest="out_dir", type=str, default="MFVLR_Dataset/images/EFS", help="Output directory for generated images")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index for parallel cluster jobs")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--start-index", type=int, default=None, help="Explicit start index (overrides sharding)")
    parser.add_argument("--end-index", type=int, default=None, help="Explicit end index")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size for faster parallel generation (e.g. 4 or 8)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 half-precision on CUDA for 2x speedup")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable FP16")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion inference steps (default: 50, use 25 for 2x speedup)")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic mock samples without checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without generating files")
    parser.add_argument("--force", action="store_true", help="Force re-generation even if output images already exist")
    parser.add_argument("--drive-manifest", type=Path, default=None,
                        help="JSON manifest of StyleGAN3 indices already uploaded to Drive")
    parser.add_argument("--drive-queue-dir", type=Path, default=None,
                        help="Directory where completed StyleGAN3 images are queued for Drive upload")

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
            batch_size=args.batch_size,
            fp16=args.fp16,
            steps=args.steps,
            mock=args.mock,
            dry_run=args.dry_run,
            force=args.force,
            drive_manifest=args.drive_manifest,
            drive_queue_dir=args.drive_queue_dir,
        )
        return 0
    except Exception as e:
        logger.error("EFS generation failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
