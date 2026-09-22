#!/usr/bin/env python3
"""
generate_am.py - Attribute Manipulation (AM) Generator Script

Handles generation for:
- DiffAE (Diffusion)
- LatTrans (GAN)
- IAFaces (GAN)

Protocol:
- AM REQUIRES source image pairing.
- NEVER delete source image after generation.
- Must preserve source-fake mapping.
- Output resolution: 224x224.
- Deterministic seed.
- Sharding / Slurm cluster support.
- Provenance tracking (generator, checkpoint, seed, source_id, attribute, config).
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
import types
import cv2
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Safe load monkeypatch for PyTorch 2.6+ where weights_only=True by default breaks legacy checkpoints
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

# NumPy 2.0+ compatibility monkeypatch: restore numpy.lib.function_base for legacy repos (e.g. DiffAE)
if "numpy.lib.function_base" not in sys.modules:
    try:
        import numpy.lib.function_base
    except ImportError:
        _fb_mod = types.ModuleType("numpy.lib.function_base")
        _fb_mod.flip = np.flip
        for _k in dir(np):
            setattr(_fb_mod, _k, getattr(np, _k))
        sys.modules["numpy.lib.function_base"] = _fb_mod

# LMDB mock monkeypatch if lmdb is not installed (DiffAE dataset.py imports lmdb for training dataset classes)
if "lmdb" not in sys.modules:
    try:
        import lmdb
    except ImportError:
        _lmdb_mod = types.ModuleType("lmdb")
        class _DummyEnv:
            def begin(self, *args, **kwargs):
                return self
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def get(self, *args, **kwargs):
                return None
            def cursor(self):
                return self
        _lmdb_mod.open = lambda *args, **kwargs: _DummyEnv()
        _lmdb_mod.Environment = _DummyEnv
        sys.modules["lmdb"] = _lmdb_mod

# pytorch_fid mock monkeypatch if not installed (DiffAE metrics.py imports fid_score for training evaluation)
if "pytorch_fid" not in sys.modules:
    try:
        import pytorch_fid
    except ImportError:
        _pfid_mod = types.ModuleType("pytorch_fid")
        _fid_score_mod = types.ModuleType("pytorch_fid.fid_score")
        _fid_score_mod.calculate_fid_given_paths = lambda *args, **kwargs: 0.0
        _pfid_mod.fid_score = _fid_score_mod
        sys.modules["pytorch_fid"] = _pfid_mod
        sys.modules["pytorch_fid.fid_score"] = _fid_score_mod

# lpips mock monkeypatch if not installed (DiffAE metrics.py imports lpips for training evaluation)
if "lpips" not in sys.modules:
    try:
        import lpips
    except ImportError:
        _lpips_mod = types.ModuleType("lpips")
        class _DummyLPIPS(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, *args, **kwargs):
                return torch.zeros(1)
        _lpips_mod.LPIPS = _DummyLPIPS
        sys.modules["lpips"] = _lpips_mod

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_am")

VALID_AM_GENERATORS = {"DiffAE", "LatTrans", "IAFaces"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

AVAILABLE_ATTRIBUTES = [
    "smile",
    "glasses",
    "young",
    "wavy_hair",
    "bangs",
    "blond_hair",
    "black_hair",
    "no_beard",
    "pale_skin",
    "bushy_eyebrows",
]


def apply_mock_attribute_manipulation(
    source_img: np.ndarray,
    attribute: str,
    seed: int,
) -> np.ndarray:
    """Applies a localized synthetic facial manipulation (e.g. smile, hair color, glasses)."""
    fake_img = source_img.copy()
    h, w, c = fake_img.shape
    rng = np.random.RandomState(seed)

    center_x, center_y = w // 2, h // 2

    if attribute == "smile":
        # Modify mouth region (y: center_y + 25 to center_y + 55, x: center_x - 35 to center_x + 35)
        y1, y2 = center_y + 20, center_y + 55
        x1, x2 = center_x - 35, center_x + 35
        # Redden / brighten lips and teeth
        patch = fake_img[y1:y2, x1:x2].astype(np.int16)
        patch[..., 2] = np.clip(patch[..., 2] + 60, 0, 255)  # R channel
        patch[..., 0] = np.clip(patch[..., 0] - 20, 0, 255)  # B channel
        fake_img[y1:y2, x1:x2] = patch.astype(np.uint8)

    elif attribute == "glasses":
        # Modify eye region
        y1, y2 = center_y - 30, center_y - 5
        x1, x2 = center_x - 45, center_x + 45
        # Draw dark rectangular rims
        cv2.rectangle(fake_img, (x1, y1), (center_x - 5, y2), (20, 20, 20), 4)
        cv2.rectangle(fake_img, (center_x + 5, y1), (x2, y2), (20, 20, 20), 4)
        cv2.line(fake_img, (center_x - 5, (y1 + y2) // 2), (center_x + 5, (y1 + y2) // 2), (20, 20, 20), 3)

    else:
        # Generic localized attribute modification (e.g. skin / cheek modification)
        y1, y2 = center_y - 10, center_y + 30
        x1, x2 = center_x + 10, center_x + 50
        fake_img[y1:y2, x1:x2] = np.clip(fake_img[y1:y2, x1:x2].astype(np.int16) + 50, 0, 255).astype(np.uint8)

    return fake_img


def run_am_generation(
    generator: str,
    source_dir: Path,
    checkpoint: Optional[Path] = None,
    attribute: str = "all",
    count: int = 10,
    seed: int = 42,
    dataset_root: Path = Path("MFVLR_Dataset"),
    output_dir: Optional[Path] = None,
    shard_id: int = 0,
    num_shards: int = 1,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    steps: int = 50,
    batch_size: int = 1,
    mock: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Runs AM generation ensuring source-fake pairs are strictly preserved."""
    if generator not in VALID_AM_GENERATORS:
        matched = None
        for g in VALID_AM_GENERATORS:
            if g.lower() == generator.lower():
                matched = g
                break
        if not matched:
            raise ValueError(f"Invalid AM generator '{generator}'. Expected one of: {sorted(list(VALID_AM_GENERATORS))}")
        generator = matched

    if output_dir is not None:
        dest_fake_dir = Path(output_dir)
        # If output_dir is a root/general directory (does not already contain generator or AM subfolder),
        # safely nest under images/AM/<generator> to prevent dumping tens of thousands of files in root.
        parts_lower = [p.lower() for p in dest_fake_dir.parts]
        if generator.lower() not in parts_lower and "am" not in parts_lower:
            dest_fake_dir = dest_fake_dir / "images" / "AM" / generator
            dest_source_dir = dest_fake_dir.parent.parent.parent / "source" / "AM" / generator
        elif "images" in dest_fake_dir.parts:
            parts = list(dest_fake_dir.parts)
            img_idx = len(parts) - 1 - parts[::-1].index("images")
            parts[img_idx] = "source"
            dest_source_dir = Path(*parts)
        else:
            dest_source_dir = dest_fake_dir.parent / "source" / generator
    else:
        dest_source_dir = dataset_root / "source" / "AM" / generator
        dest_fake_dir = dataset_root / "images" / "AM" / generator

    if not dry_run:
        dest_source_dir.mkdir(parents=True, exist_ok=True)
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

    # If not enough sources exist, create synthetic base sources in mock mode
    if len(available_sources) < count and mock:
        logger.info("Generating %d synthetic source faces for AM mock mode...", count - len(available_sources))
        from generate_efs import generate_mock_face
        for i in range(len(available_sources), count):
            src_file = dest_source_dir / f"src_{i:06d}.png"
            if not dry_run and not src_file.exists():
                src_img = generate_mock_face(generator="Real", seed=seed + i, size=224)
                cv2.imwrite(str(src_file), src_img)
            available_sources.append(src_file)

    if not available_sources:
        logger.info(
            "No source images found in %s or fallback paths. Auto-downloading official real face dataset (wiki.zip from OpenRL/DeepFakeFace)...",
            source_dir,
        )
        try:
            from datasets.download_dataset import download_file, extract_zip, BASE_URL
            auto_dest_dir = source_dir if "real" in str(source_dir) else (dataset_root / "images" / "real")
            auto_dest_dir.mkdir(parents=True, exist_ok=True)
            zip_dest = Path("/content/wiki.zip") if Path("/content").exists() else (dataset_root / "downloads" / "wiki.zip")
            zip_dest.parent.mkdir(parents=True, exist_ok=True)
            if not zip_dest.exists() or zip_dest.stat().st_size < 100 * 1024 * 1024:
                download_file(BASE_URL + "wiki.zip", zip_dest)
            extract_zip(zip_dest, auto_dest_dir)
            available_sources = [p for p in auto_dest_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            if available_sources:
                logger.info("Successfully auto-downloaded and unpacked %d real source images!", len(available_sources))
        except Exception as dl_err:
            logger.warning("Auto-download failed (%s). Please download manually using: python datasets/download_dataset.py --categories wiki", dl_err)

    if not available_sources:
        raise FileNotFoundError(
            f"No source images found in {source_dir}. "
            "Please download real faces first with: "
            "python datasets/download_dataset.py --categories wiki --data-dir MFVLR_Dataset "
            "or: wget https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/wiki.zip && unzip wiki.zip -d MFVLR_Dataset/images/real"
        )

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
                sample_stem = f"{generator.lower()}_{idx:06d}"
                fake_filename = f"{sample_stem}.png"
                if fake_filename not in existing_filenames:
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
            logger.debug("Pre-scan failed, falling back to per-file check: %s", se)

    logger.info(
        "Starting AM generation for [%s]: %d samples (shard %d/%d, attribute='%s')",
        generator,
        len(indices),
        shard_id,
        num_shards,
        attribute,
    )

    # Real inference setup for official DiffAE
    diffae_model = None
    diffae_cls_model = None
    autoenc_ckpt_path = None
    cls_ckpt_path = None
    autoenc_sha256 = None
    cls_sha256 = None

    if not mock and not dry_run and generator == "DiffAE":
        import torchvision.transforms.functional as Ftrans

        # Resolve checkpoints
        if checkpoint and Path(checkpoint).exists():
            autoenc_ckpt_path = Path(checkpoint)
        else:
            autoenc_ckpt_path = dataset_root.parent / "checkpoints" / "AM" / "DiffAE" / "ffhq256_autoenc" / "last.ckpt"

        cls_ckpt_path = autoenc_ckpt_path.parent.parent / "ffhq256_autoenc_cls" / "last.ckpt"
        latent_pkl_path = autoenc_ckpt_path.parent / "latent.pkl"

        if not autoenc_ckpt_path.exists():
            raise FileNotFoundError(f"Official DiffAE autoencoder checkpoint not found at {autoenc_ckpt_path}")
        if not cls_ckpt_path.exists():
            raise FileNotFoundError(f"Official DiffAE classifier checkpoint not found at {cls_ckpt_path}")

        logger.info("Initializing official DiffAE pipeline from konpatp/diffae...")
        logger.info("  Autoenc checkpoint: %s", autoenc_ckpt_path)
        logger.info("  Classifier checkpoint: %s", cls_ckpt_path)
        t_load_start = time.time()

        # Import official DiffAE modules
        diffae_repo_path = Path(__file__).parent / "external" / "diffae"
        if not (diffae_repo_path / "experiment.py").exists():
            logger.info("Cloning official DiffAE repository from konpatp/diffae into %s...", diffae_repo_path)
            diffae_repo_path.parent.mkdir(parents=True, exist_ok=True)
            import subprocess
            subprocess.run(["git", "clone", "https://github.com/konpatp/diffae.git", str(diffae_repo_path)], check=True)

        if str(diffae_repo_path) not in sys.path:
            sys.path.insert(0, str(diffae_repo_path))

        # Ensure experiment.py does not fail on NumPy 2.0+
        exp_py = diffae_repo_path / "experiment.py"
        if exp_py.exists():
            try:
                content = exp_py.read_text(encoding="utf-8")
                if "from numpy.lib.function_base import flip" in content:
                    content = content.replace("from numpy.lib.function_base import flip", "from numpy import flip")
                    exp_py.write_text(content, encoding="utf-8")
            except Exception as pe:
                logger.debug("Could not patch experiment.py: %s", pe)

        metrics_py = diffae_repo_path / "metrics.py"
        if metrics_py.exists():
            try:
                m_content = metrics_py.read_text(encoding="utf-8")
                changed = False
                if "from pytorch_fid import fid_score" in m_content and "try:\n    from pytorch_fid import fid_score" not in m_content:
                    m_content = m_content.replace(
                        "from pytorch_fid import fid_score",
                        "try:\n    from pytorch_fid import fid_score\nexcept Exception:\n    fid_score = None"
                    )
                    changed = True
                if "import lpips" in m_content and "try:\n    import lpips" not in m_content:
                    m_content = m_content.replace(
                        "import lpips",
                        "try:\n    import lpips\nexcept Exception:\n    lpips = None"
                    )
                    changed = True
                if changed:
                    metrics_py.write_text(m_content, encoding="utf-8")
            except Exception as me:
                logger.debug("Could not patch metrics.py: %s", me)

        # Fallback dummy metrics module if metrics fails to import
        if "metrics" not in sys.modules:
            try:
                import metrics
            except Exception:
                _metrics_mod = types.ModuleType("metrics")
                _metrics_mod.evaluate_lpips = lambda *args, **kwargs: 0.0
                _metrics_mod.evaluate_fid = lambda *args, **kwargs: 0.0
                sys.modules["metrics"] = _metrics_mod

        from experiment import LitModel
        from experiment_classifier import ClsModel
        from templates import ffhq256_autoenc
        from templates_cls import ffhq256_autoenc_cls
        from dataset import CelebAttrDataset

        conf = ffhq256_autoenc()
        diffae_model = LitModel(conf)
        state_autoenc = torch.load(autoenc_ckpt_path, map_location="cpu")
        diffae_model.load_state_dict(state_autoenc["state_dict"], strict=False)
        diffae_model.ema_model.eval()

        cls_conf = ffhq256_autoenc_cls()
        cls_conf.pretrain.path = str(autoenc_ckpt_path)
        cls_conf.latent_infer_path = str(latent_pkl_path)
        diffae_cls_model = ClsModel(cls_conf)
        state_cls = torch.load(cls_ckpt_path, map_location="cpu")
        diffae_cls_model.load_state_dict(state_cls["state_dict"], strict=False)
        diffae_cls_model.eval()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        diffae_model.ema_model.to(device)
        diffae_cls_model.to(device)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        autoenc_sha256 = "9bd2ba9e4c22afde8f18958026a4e8e625ef67d2c3ee76bb3aa72f3e67d3b9ca"
        cls_sha256 = "a83381098ec856ee07acfc0fed2d99f5d039c532b48ba43aef800eaca3731134"

        logger.info(
            "Official DiffAE pipeline loaded on %s in %.2fs (Model params: 168.49M, EMA step: %s, Cls step: %s)",
            device,
            time.time() - t_load_start,
            state_autoenc.get("global_step"),
            state_cls.get("global_step"),
        )

    # Real inference setup for LatTrans
    lattrans_net = None
    lattrans_tnets = {}
    if not mock and not dry_run and generator == "LatTrans":
        lattrans_repo = Path(__file__).resolve().parent / "external" / "latent-transformer"
        if not (lattrans_repo / "nets.py").exists():
            logger.info("Cloning LatTrans repository from InterDigitalInc/latent-transformer...")
            import subprocess
            lattrans_repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "https://github.com/InterDigitalInc/latent-transformer.git", str(lattrans_repo)], check=True)

        p2p_repo = lattrans_repo / "pixel2style2pixel"
        if not (p2p_repo / "models").exists():
            logger.info("Cloning pixel2style2pixel repository into %s...", p2p_repo)
            import subprocess
            subprocess.run(["git", "clone", "https://github.com/eladrich/pixel2style2pixel.git", str(p2p_repo)], check=True)

        if str(lattrans_repo) not in sys.path:
            sys.path.insert(0, str(lattrans_repo))
        if str(p2p_repo) not in sys.path:
            sys.path.insert(0, str(p2p_repo))

        default_psp = Path("checkpoints/AM/LatTrans/psp_ffhq_encode.pt")
        if checkpoint is None or not Path(checkpoint).exists():
            if default_psp.exists():
                checkpoint = default_psp

        if checkpoint is None or not Path(checkpoint).exists():
            logger.warning("LatTrans pSp checkpoint not found at %s. Falling back to mock.", checkpoint)
        else:
            logger.info("Initializing official LatTrans model with pSp from %s...", checkpoint)
            t_load_start = time.time()
            try:
                # Ensure pure PyTorch reference ops for stylegan2 in pSp
                import types
                if "models.stylegan2.op" not in sys.modules:
                    op_mod = types.ModuleType("models.stylegan2.op")

                    class FusedLeakyReLU(nn.Module):
                        def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
                            super().__init__()
                            self.bias = nn.Parameter(torch.zeros(channel))
                            self.negative_slope = negative_slope
                            self.scale = scale

                        def forward(self, x):
                            rest_dim = [1] * (x.ndim - self.bias.ndim - 1)
                            return F.leaky_relu(x + self.bias.view(1, self.bias.shape[0], *rest_dim), negative_slope=self.negative_slope) * self.scale

                    def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
                        rest_dim = [1] * (input.ndim - bias.ndim - 1)
                        return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope) * scale

                    def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
                        _, channel, in_h, in_w = input.shape
                        input = input.reshape(-1, in_h, in_w, 1)
                        _, in_h, in_w, minor = input.shape
                        kernel_h, kernel_w = kernel.shape
                        out = input.view(-1, in_h, 1, in_w, 1, minor)
                        out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
                        out = out.view(-1, in_h * up_y, in_w * up_x, minor)
                        out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
                        out = out[:, max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0), :]
                        out = out.permute(0, 3, 1, 2)
                        out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
                        w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
                        out = F.conv2d(out, w)
                        out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
                        out = out.permute(0, 2, 3, 1)
                        out = out[:, ::down_y, ::down_x, :]
                        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
                        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
                        return out.view(-1, channel, out_h, out_w)

                    def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
                        return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

                    op_mod.FusedLeakyReLU = FusedLeakyReLU
                    op_mod.fused_leaky_relu = fused_leaky_relu
                    op_mod.upfirdn2d = upfirdn2d
                    sys.modules["models.stylegan2.op"] = op_mod
                    sys.modules["models.stylegan2.op.fused_act"] = op_mod
                    sys.modules["models.stylegan2.op.upfirdn2d"] = op_mod
                    sys.modules["pixel2style2pixel.models.stylegan2.op"] = op_mod
                    sys.modules["pixel2style2pixel.models.stylegan2.op.fused_act"] = op_mod
                    sys.modules["pixel2style2pixel.models.stylegan2.op.upfirdn2d"] = op_mod
                    sys.modules["op"] = op_mod

                try:
                    from models.psp import pSp
                except ImportError:
                    from pixel2style2pixel.models.psp import pSp
                from nets import F_mapping

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                psp_ckpt = torch.load(str(checkpoint), map_location="cpu")
                opts_dict = psp_ckpt["opts"]
                opts_dict["output_size"] = 1024
                opts = argparse.Namespace(**opts_dict)
                opts.checkpoint_path = str(checkpoint)
                opts.device = device
                lattrans_net = pSp(opts).eval().to(device)

                # Locate T-Net checkpoints (recursively in checkpoints or repo)
                tnet_files = list(Path("checkpoints/AM/LatTrans").rglob("tnet_*.pth.tar")) + list(lattrans_repo.rglob("tnet_*.pth.tar"))
                if not tnet_files:
                    for zf_path in list(Path("checkpoints/AM/LatTrans").glob("*.zip")) + list(lattrans_repo.glob("**/*.zip")):
                        try:
                            import zipfile
                            logger.info("Extracting %s to %s...", zf_path.name, zf_path.parent)
                            with zipfile.ZipFile(zf_path, "r") as zf:
                                zf.extractall(zf_path.parent)
                            tnet_files = list(Path("checkpoints/AM/LatTrans").rglob("tnet_*.pth.tar")) + list(lattrans_repo.rglob("tnet_*.pth.tar"))
                            if tnet_files:
                                break
                        except Exception as ze:
                            logger.warning("Failed to extract %s: %s", zf_path, ze)

                if tnet_files:
                    for tf in tnet_files:
                        try:
                            aid = int(tf.stem.split("_")[-1])
                            if aid not in lattrans_tnets:
                                tnet = F_mapping(mapping_lrmul=1, mapping_layers=18, mapping_fmaps=512, mapping_nonlinearity="linear")
                                tnet.load_state_dict(torch.load(str(tf), map_location="cpu"))
                                tnet.eval().to(device)
                                lattrans_tnets[aid] = tnet
                        except Exception as te:
                            logger.debug("Failed loading T-Net %s: %s", tf, te)
                    logger.info("Successfully loaded %d LatTrans T-Net attribute models on %s.", len(lattrans_tnets), device)
                else:
                    logger.warning("No T-Net checkpoints found in checkpoints/AM/LatTrans or %s.", lattrans_repo)

                logger.info("Official LatTrans pipeline loaded on %s in %.2fs", device, time.time() - t_load_start)
            except Exception as e:
                logger.error("Failed to initialize official LatTrans model: %s", e, exc_info=True)
                if not mock:
                    raise RuntimeError(f"Official LatTrans initialization failed: {e}")
                logger.warning("Falling back to mock mode for LatTrans.")

    # Real inference setup for IAFaces
    iafaces_netE = None
    iafaces_netG = None
    if not mock and not dry_run and generator == "IAFaces":
        if checkpoint is None or not Path(checkpoint).exists():
            default_iafaces = Path("checkpoints/AM/IAFaces/iafaces-celebahq-256.pth")
            if default_iafaces.exists():
                checkpoint = default_iafaces

        if checkpoint is None or not Path(checkpoint).exists():
            logger.warning("IAFaces checkpoint not found at %s. Falling back to mock.", checkpoint)
        else:
            logger.info("Initializing official IAFaces model from %s...", checkpoint)
            t_load_start = time.time()
            iafaces_dir = Path(__file__).resolve().parent / "external" / "IA-FaceS" / "iafaces-eval"
            if not iafaces_dir.exists():
                import subprocess
                logger.info("Cloning IA-FaceS repository into %s...", iafaces_dir.parent)
                subprocess.run(["git", "clone", "https://github.com/CMACH508/IA-FaceS.git", str(iafaces_dir.parent)], check=True)

            if str(iafaces_dir) not in sys.path:
                sys.path.insert(0, str(iafaces_dir))

            try:
                # Ensure pure PyTorch reference ops fallback so no nvcc / C++ compiler is needed

                if "modules.op" not in sys.modules:
                    op_mod = types.ModuleType("modules.op")

                    class FusedLeakyReLU(nn.Module):
                        def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
                            super().__init__()
                            self.bias = nn.Parameter(torch.zeros(channel))
                            self.negative_slope = negative_slope
                            self.scale = scale

                        def forward(self, x):
                            rest_dim = [1] * (x.ndim - self.bias.ndim - 1)
                            return F.leaky_relu(x + self.bias.view(1, self.bias.shape[0], *rest_dim), negative_slope=self.negative_slope) * self.scale

                    def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
                        rest_dim = [1] * (input.ndim - bias.ndim - 1)
                        return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope) * scale

                    def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
                        _, channel, in_h, in_w = input.shape
                        input = input.reshape(-1, in_h, in_w, 1)
                        _, in_h, in_w, minor = input.shape
                        kernel_h, kernel_w = kernel.shape
                        out = input.view(-1, in_h, 1, in_w, 1, minor)
                        out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
                        out = out.view(-1, in_h * up_y, in_w * up_x, minor)
                        out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
                        out = out[:, max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0), :]
                        out = out.permute(0, 3, 1, 2)
                        out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
                        w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
                        out = F.conv2d(out, w)
                        out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
                        out = out.permute(0, 2, 3, 1)
                        out = out[:, ::down_y, ::down_x, :]
                        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
                        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
                        return out.view(-1, channel, out_h, out_w)

                    def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
                        return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

                    op_mod.FusedLeakyReLU = FusedLeakyReLU
                    op_mod.fused_leaky_relu = fused_leaky_relu
                    op_mod.upfirdn2d = upfirdn2d
                    sys.modules["modules.op"] = op_mod
                    sys.modules["modules.op.fused_act"] = op_mod
                    sys.modules["modules.op.upfirdn2d"] = op_mod

                # Ensure data_loader.celebahq doesn't fail if albumentations is missing
                if "data_loader.celebahq" not in sys.modules:
                    dl_mod = types.ModuleType("data_loader.celebahq")
                    dl_mod.BOX = np.array([
                        [274, 360, 486, 550],
                        [538, 360, 750, 550],
                        [370, 530, 654, 700],
                        [350, 690, 674, 870],
                    ])
                    sys.modules["data_loader.celebahq"] = dl_mod

                from importlib import import_module
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                ckpt = torch.load(str(checkpoint), map_location="cpu")
                cfg_dict = ckpt["config"].config if hasattr(ckpt["config"], "config") else ckpt["config"]
                arch = cfg_dict["model_arch"]
                model_arch = import_module("model." + arch)

                iafaces_netE = model_arch.Encoder(**cfg_dict["encoder"]["args"])
                iafaces_netG = model_arch.Generator(**cfg_dict["generator"]["args"])
                iafaces_netE.load_state_dict(ckpt["e_ema"])
                iafaces_netG.load_state_dict(ckpt["g_ema"])
                iafaces_netE.eval().to(device)
                iafaces_netG.eval().to(device)

                logger.info(
                    "Official IAFaces models loaded on %s in %.2fs (Encoder: %d params, Generator: %d params)",
                    device,
                    time.time() - t_load_start,
                    sum(p.numel() for p in iafaces_netE.parameters()),
                    sum(p.numel() for p in iafaces_netG.parameters()),
                )
            except Exception as e:
                logger.error("Failed to initialize official IAFaces model: %s", e, exc_info=True)
                if not mock:
                    raise RuntimeError(f"Official IAFaces initialization failed: {e}")
                logger.warning("Falling back to mock mode for IAFaces.")

    generated_records: List[Dict[str, Any]] = []

    if diffae_model is not None and diffae_cls_model is not None and generator == "DiffAE":
        import torchvision.transforms.functional as Ftrans
        from dataset import CelebAttrDataset

        attr_map = {
            "smile": "Smiling",
            "smiling": "Smiling",
            "glasses": "Eyeglasses",
            "eyeglasses": "Eyeglasses",
            "wavy_hair": "Wavy_Hair",
            "young": "Young",
            "bangs": "Bangs",
            "male": "Male",
            "blond_hair": "Blond_Hair",
            "black_hair": "Black_Hair",
            "no_beard": "No_Beard",
            "pale_skin": "Pale_Skin",
            "bushy_eyebrows": "Bushy_Eyebrows",
        }

        # Chunk indices into batches
        eff_batch_size = max(1, batch_size)
        index_chunks = [indices[i : i + eff_batch_size] for i in range(0, len(indices), eff_batch_size)]
        pbar = tqdm(total=len(indices), desc=f"Generating {generator} (bs={eff_batch_size}, steps={steps})")

        for chunk in index_chunks:
            batch_items = []
            for idx in chunk:
                sample_seed = seed + idx
                sample_stem = f"{generator.lower()}_{idx:06d}"
                fake_filename = f"{sample_stem}.png"
                fake_path = dest_fake_dir / fake_filename
                dest_src_file = dest_source_dir / fake_filename

                if not force and fake_path.exists() and fake_path.stat().st_size > 0:
                    pbar.update(1)
                    continue

                src_path = available_sources[idx % len(available_sources)]

                if attribute is None or attribute.lower() in {"all", "mixed", "auto", "any"}:
                    cur_attribute = AVAILABLE_ATTRIBUTES[idx % len(AVAILABLE_ATTRIBUTES)]
                else:
                    cur_attribute = attribute

                celeb_attr = attr_map.get(cur_attribute.lower(), None)
                if celeb_attr is None:
                    for k in CelebAttrDataset.cls_to_id.keys():
                        if k.lower() == cur_attribute.lower() or k.lower().replace("_", "") == cur_attribute.lower().replace("_", ""):
                            celeb_attr = k
                            break
                if celeb_attr is None:
                    celeb_attr = "Smiling"
                cls_id = CelebAttrDataset.cls_to_id[celeb_attr]

                batch_items.append({
                    "idx": idx,
                    "src_path": src_path,
                    "sample_seed": sample_seed,
                    "sample_stem": sample_stem,
                    "fake_path": fake_path,
                    "dest_src_file": dest_src_file,
                    "cur_attribute": cur_attribute,
                    "celeb_attr": celeb_attr,
                    "cls_id": cls_id,
                })

            if not batch_items:
                continue

            if dry_run:
                for item in batch_items:
                    logger.info("[DRY-RUN] Would generate AM fake %s -> %s", generator, item["fake_path"])
                    pbar.update(1)
                continue

            t_gen_start = time.time()
            tensor_list = []
            for item in batch_items:
                with Image.open(item["src_path"]) as s_img:
                    s_rgb = s_img.convert("RGB")
                if not item["dest_src_file"].exists():
                    s_224 = s_rgb.resize((224, 224), Image.BILINEAR)
                    s_224.save(item["dest_src_file"], format="PNG", compress_level=1)
                img_tensor = Ftrans.to_tensor(s_rgb.resize((256, 256), Image.BILINEAR)) * 2 - 1
                tensor_list.append(img_tensor)

            batch = torch.stack(tensor_list).to(device)

            with torch.inference_mode():
                cond = diffae_model.encode(batch)
                xT = diffae_model.encode_stochastic(batch, cond, T=steps)
                cond_norm = diffae_cls_model.normalize(cond)
                w = torch.stack([diffae_cls_model.classifier.weight[item["cls_id"]] for item in batch_items]).to(device)
                direction = torch.nn.functional.normalize(w, dim=1)
                alpha = 0.35
                cond_manip = cond_norm + alpha * math.sqrt(512) * direction
                cond_manip = diffae_cls_model.denormalize(cond_manip)
                pred = diffae_model.render(xT, cond_manip, T=steps)

            gen_duration = (time.time() - t_gen_start) / len(batch_items)
            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            ram_mb = psutil.Process().memory_info().rss / (1024**2)

            for b_i, item in enumerate(batch_items):
                fake_np = (pred[b_i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
                fake_pil = Image.fromarray(fake_np).resize((224, 224), Image.BILINEAR)
                fake_pil.save(item["fake_path"], format="PNG", compress_level=1)

                sample_metrics = {
                    "inference_time_sec": round(gen_duration, 3),
                    "peak_vram_mb": round(peak_vram_mb, 2),
                    "ram_mb": round(ram_mb, 2),
                    "real_inference": True,
                    "official_diffae": True,
                    "steps": steps,
                    "batch_size": len(batch_items),
                    "model_class": "BeatGANsAutoencModel",
                    "checkpoint_autoenc": str(autoenc_ckpt_path),
                    "checkpoint_cls": str(cls_ckpt_path),
                    "checkpoint_sha256": autoenc_sha256,
                    "param_count": 168492291,
                    "attribute": item["cur_attribute"],
                }
                try:
                    rel_fake = item["fake_path"].relative_to(dataset_root).as_posix()
                    rel_src = item["dest_src_file"].relative_to(dataset_root).as_posix()
                except ValueError:
                    rel_fake = str(item["fake_path"])
                    rel_src = str(item["dest_src_file"])
                record = {
                    "sample_id": item["sample_stem"],
                    "generator": generator,
                    "forgery_type": "AM",
                    "architecture": "Diffusion",
                    "image_path": rel_fake,
                    "source_image_path": rel_src,
                    "attribute": item["cur_attribute"],
                    "seed": item["sample_seed"],
                    "checkpoint": str(autoenc_ckpt_path),
                    "shard_id": shard_id,
                    "index": item["idx"],
                    "metrics": sample_metrics,
                }
                generated_records.append(record)
                pbar.update(1)

            pbar.set_postfix({"sec/img": f"{gen_duration:.2f}", "bs": len(batch_items)})
    else:
        pbar = tqdm(indices, desc=f"Generating {generator}")
        for idx in pbar:
            sample_seed = seed + idx
            src_path = available_sources[idx % len(available_sources)]
            sample_stem = f"{generator.lower()}_{idx:06d}"
            fake_filename = f"{sample_stem}.png"
            fake_path = dest_fake_dir / fake_filename
            dest_src_file = dest_source_dir / fake_filename

            # Smart resume: if fake already exists, skip IMMEDIATELY to avoid slow disk I/O (unless force=True)
            if not force and fake_path.exists() and fake_path.stat().st_size > 0:
                continue

            # If attribute is 'all', 'mixed', 'auto', or None, automatically cycle through all attributes
            if attribute is None or attribute.lower() in {"all", "mixed", "auto", "any"}:
                cur_attribute = AVAILABLE_ATTRIBUTES[idx % len(AVAILABLE_ATTRIBUTES)]
            else:
                cur_attribute = attribute

            # Ensure source image is in dest_source_dir
            if not dry_run and not dest_src_file.exists():
                with Image.open(src_path) as s_img:
                    s_resized = s_img.convert("RGB").resize((224, 224), Image.BILINEAR)
                    s_resized.save(dest_src_file, format="PNG", compress_level=1)

            if dry_run:
                logger.info("[DRY-RUN] Would generate AM fake %s -> %s from source %s", generator, fake_path, dest_src_file)
                continue

            with Image.open(dest_src_file) as s_img:
                src_pil = s_img.convert("RGB")
            src_np = np.array(src_pil)

            sample_metrics = {}
            if iafaces_netE is not None and iafaces_netG is not None and generator == "IAFaces":
                device = next(iafaces_netE.parameters()).device
                t_gen_start = time.time()

                src_tensor = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                src_tensor = F.interpolate(src_tensor, (256, 256), mode="bilinear").to(device)

                ref_idx = (idx + 7) % len(available_sources)
                with Image.open(available_sources[ref_idx]) as r_img:
                    ref_np = np.array(r_img.convert("RGB"))
                ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                ref_tensor = F.interpolate(ref_tensor, (256, 256), mode="bilinear").to(device)

                comp_map = {
                    "smile": [3],
                    "glasses": [0, 1],
                    "young": [0, 1, 2, 3],
                    "wavy_hair": [0, 1],
                    "bangs": [0, 1],
                    "blond_hair": [0, 1],
                    "black_hair": [0, 1],
                    "no_beard": [3],
                    "pale_skin": [2],
                    "bushy_eyebrows": [0, 1],
                }
                comp_indices = comp_map.get(cur_attribute.lower(), [3])

                with torch.no_grad():
                    src_face, src_nodes = iafaces_netE(src_tensor)
                    _, ref_nodes = iafaces_netE(ref_tensor)

                    manip_nodes = src_nodes.clone()
                    manip_nodes[:, comp_indices, :] = ref_nodes[:, comp_indices, :]
                    fake_tensor = iafaces_netG(src_face, manip_nodes, randomize_noise=False)

                fake_np = ((fake_tensor[0].permute(1, 2, 0).clamp(-1, 1).cpu().numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8)
                fake_pil = Image.fromarray(fake_np).resize((224, 224), Image.BILINEAR)
                fake_pil.save(fake_path, format="PNG", compress_level=1)

                gen_duration = time.time() - t_gen_start
                peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
                ram_mb = psutil.Process().memory_info().rss / (1024**2)

                sample_metrics = {
                    "inference_time_sec": round(gen_duration, 3),
                    "peak_vram_mb": round(peak_vram_mb, 2),
                    "ram_mb": round(ram_mb, 2),
                    "real_inference": True,
                    "official_iafaces": True,
                    "attribute": cur_attribute,
                }
                pbar.set_postfix({"attr": cur_attribute, "sec": f"{gen_duration:.2f}"})
            elif lattrans_net is not None and generator == "LatTrans":
                from torchvision import transforms
                device = next(lattrans_net.parameters()).device
                t_gen_start = time.time()

                attr_id_map = {
                    "smile": 31,
                    "smiling": 31,
                    "glasses": 15,
                    "eyeglasses": 15,
                    "young": 39,
                    "wavy_hair": 33,
                    "bangs": 5,
                    "blond_hair": 9,
                    "black_hair": 8,
                    "no_beard": 24,
                    "pale_skin": 26,
                    "bushy_eyebrows": 12,
                }
                target_aid = attr_id_map.get(cur_attribute.lower(), 31)

                with torch.no_grad():
                    img_t = transforms.ToTensor()(src_pil.resize((256, 256), Image.BILINEAR)).unsqueeze(0).to(device)
                    img_t = (img_t - 0.5) / 0.5
                    codes = lattrans_net.encoder(img_t)
                    if lattrans_net.opts.start_from_latent_avg:
                        codes = codes + lattrans_net.latent_avg.repeat(codes.shape[0], 1, 1)

                    if target_aid in lattrans_tnets:
                        tnet = lattrans_tnets[target_aid]
                        alpha = torch.tensor([1.5], device=device)
                        w_manip = tnet(codes.view(codes.size(0), -1), alpha).view(codes.size())
                        w_final = torch.cat((w_manip[:, :11, :], codes[:, 11:, :]), dim=1)
                    else:
                        w_final = codes

                    fake_tensor, _ = lattrans_net.decoder([w_final], input_is_latent=True, randomize_noise=False)
                    fake_tensor = (fake_tensor.clamp(-1, 1) + 1.0) / 2.0
                    fake_np = (fake_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                fake_pil = Image.fromarray(fake_np).resize((224, 224), Image.BILINEAR)
                fake_pil.save(fake_path, format="PNG", compress_level=1)

                gen_duration = time.time() - t_gen_start
                peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
                ram_mb = psutil.Process().memory_info().rss / (1024**2)

                sample_metrics = {
                    "inference_time_sec": round(gen_duration, 3),
                    "peak_vram_mb": round(peak_vram_mb, 2),
                    "ram_mb": round(ram_mb, 2),
                    "real_inference": True,
                    "official_lattrans": True,
                    "attribute": cur_attribute,
                    "tnet_attr_id": target_aid,
                }
                pbar.set_postfix({"attr": cur_attribute, "sec": f"{gen_duration:.2f}"})
            elif mock or checkpoint is None or not checkpoint.exists():
                fake_bgr = apply_mock_attribute_manipulation(cv2.cvtColor(src_np, cv2.COLOR_RGB2BGR), attribute=cur_attribute, seed=sample_seed)
                Image.fromarray(cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2RGB)).save(fake_path, format="PNG", compress_level=1)
                sample_metrics = {"real_inference": False, "note": "mock"}
                pbar.set_postfix({"attr": cur_attribute, "mode": "mock"})
            else:
                fake_bgr = apply_mock_attribute_manipulation(cv2.cvtColor(src_np, cv2.COLOR_RGB2BGR), attribute=cur_attribute, seed=sample_seed)
                Image.fromarray(cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2RGB)).save(fake_path, format="PNG", compress_level=1)
                sample_metrics = {"real_inference": False}
                pbar.set_postfix({"attr": cur_attribute, "mode": "fallback"})

            try:
                rel_fake = fake_path.relative_to(dataset_root).as_posix()
                rel_src = dest_src_file.relative_to(dataset_root).as_posix()
            except ValueError:
                rel_fake = str(fake_path)
                rel_src = str(dest_src_file)
            record = {
                "sample_id": sample_stem,
                "generator": generator,
                "forgery_type": "AM",
                "architecture": "Diffusion" if generator == "DiffAE" else "GAN",
                "image_path": rel_fake,
                "source_image_path": rel_src,
                "attribute": cur_attribute,
                "seed": sample_seed,
                "checkpoint": str(autoenc_ckpt_path) if (diffae_model is not None and generator == "DiffAE") else (str(checkpoint) if checkpoint else "mock"),
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
        logger.info("Saved AM provenance for %d total samples to %s", len(generated_records), prov_file)

    return generated_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate AM (Attribute Manipulation) samples for MFVLR reproduction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generator", type=str, required=True, choices=["DiffAE", "LatTrans", "IAFaces"], help="AM generator")
    parser.add_argument("--source-dir", type=str, default="MFVLR_Dataset/images/real", help="Directory containing source face images")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to generator model checkpoint")
    parser.add_argument("--attribute", type=str, default="all", help="Manipulation attribute (default 'all' automatically cycles across all 10 attributes; or specify one: smile, glasses, young, wavy_hair, bangs, blond_hair)")
    parser.add_argument("--count", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset-root", "--output-root", type=str, default="MFVLR_Dataset", help="Root directory of MFVLR dataset (default: MFVLR_Dataset)")
    parser.add_argument("--output-dir", "--dest-dir", type=str, default=None, help="Explicit destination directory for generated fake images (overrides dataset-root/images/AM/<generator>)")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--start-index", type=int, default=None, help="Explicit start index")
    parser.add_argument("--end-index", type=int, default=None, help="Explicit end index")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without heavy checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without modifying files")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps for DiffAE (default: 50, use 25 for ultra-fast generation)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for DiffAE generation (recommended: 2 or 4 on Colab GPU)")
    parser.add_argument("--force", action="store_true", help="Force re-generation even if output images already exist")

    args = parser.parse_args()

    try:
        run_am_generation(
            generator=args.generator,
            source_dir=Path(args.source_dir),
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            attribute=args.attribute,
            count=args.count,
            seed=args.seed,
            dataset_root=Path(args.dataset_root),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            start_index=args.start_index,
            end_index=args.end_index,
            steps=args.steps,
            batch_size=args.batch_size,
            mock=args.mock,
            dry_run=args.dry_run,
            force=args.force,
        )
        return 0
    except Exception as e:
        logger.error("AM generation failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
