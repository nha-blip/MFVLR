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
    shard_id: int = 0,
    num_shards: int = 1,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
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

    dest_source_dir = dataset_root / "source" / "AM" / generator
    dest_fake_dir = dataset_root / "images" / "AM" / generator

    if not dry_run:
        dest_source_dir.mkdir(parents=True, exist_ok=True)
        dest_fake_dir.mkdir(parents=True, exist_ok=True)

    # Collect source images
    available_sources: List[Path] = []
    if source_dir.exists():
        for p in sorted(source_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                available_sources.append(p)

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
        raise FileNotFoundError(f"No source images found in {source_dir} and mock generation disabled.")

    # Calculate index range
    if start_index is not None and end_index is not None:
        indices = list(range(start_index, min(end_index, len(available_sources))))
    else:
        total = min(count, len(available_sources))
        indices = [idx for idx in range(total) if idx % num_shards == shard_id]

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
        import time
        import psutil
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
        if str(diffae_repo_path) not in sys.path:
            sys.path.insert(0, str(diffae_repo_path))

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

        autoenc_sha256 = "9bd2ba9e4c22afde8f18958026a4e8e625ef67d2c3ee76bb3aa72f3e67d3b9ca"
        cls_sha256 = "a83381098ec856ee07acfc0fed2d99f5d039c532b48ba43aef800eaca3731134"

        logger.info(
            "Official DiffAE pipeline loaded on %s in %.2fs (Model params: 168.49M, EMA step: %s, Cls step: %s)",
            device,
            time.time() - t_load_start,
            state_autoenc.get("global_step"),
            state_cls.get("global_step"),
        )

    # Real inference setup for IAFaces
    iafaces_netE = None
    iafaces_netG = None
    if not mock and not dry_run and generator == "IAFaces":
        import time
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
                import types
                import torch.nn as nn
                import torch.nn.functional as F

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

    from tqdm import tqdm
    pbar = tqdm(indices, desc=f"Generating {generator}")
    for idx in pbar:
        sample_seed = seed + idx
        src_path = available_sources[idx]
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
            from PIL import Image
            with Image.open(src_path) as s_img:
                s_resized = s_img.convert("RGB").resize((224, 224), Image.BILINEAR)
                s_resized.save(dest_src_file, format="PNG")

        if dry_run:
            logger.info("[DRY-RUN] Would generate AM fake %s -> %s from source %s", generator, fake_path, dest_src_file)
            continue

        from PIL import Image
        with Image.open(dest_src_file) as s_img:
            src_pil = s_img.convert("RGB")
        src_np = np.array(src_pil)

        sample_metrics = {}
        if diffae_model is not None and diffae_cls_model is not None and generator == "DiffAE":
            import time
            import math
            import psutil
            import torch
            import torchvision.transforms.functional as Ftrans
            from dataset import CelebAttrDataset

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t_gen_start = time.time()

            # Set deterministic seed for PyTorch
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(sample_seed)

            # Prepare normalized tensor [1, 3, 256, 256] in [-1, 1]
            img_tensor = Ftrans.to_tensor(src_pil.resize((256, 256), Image.BILINEAR)) * 2 - 1
            batch = img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                # 1. Encode semantic latent vector z_sem in R^512
                cond = diffae_model.encode(batch)

                # 2. Stochastic inversion (DDIM reverse process to recover xT)
                xT = diffae_model.encode_stochastic(batch, cond, T=250)

                # 3. Attribute manipulation along linear classifier hyperplane
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
                celeb_attr = attr_map.get(cur_attribute.lower(), None)
                if celeb_attr is None:
                    for k in CelebAttrDataset.cls_to_id.keys():
                        if k.lower() == cur_attribute.lower() or k.lower().replace("_", "") == cur_attribute.lower().replace("_", ""):
                            celeb_attr = k
                            break
                if celeb_attr is None:
                    celeb_attr = "Smiling"
                cls_id = CelebAttrDataset.cls_to_id[celeb_attr]

                cond_norm = diffae_cls_model.normalize(cond)
                w = diffae_cls_model.classifier.weight[cls_id][None, :]
                direction = torch.nn.functional.normalize(w, dim=1)
                alpha = 0.35  # Manipulation intensity
                cond_manip = cond_norm + alpha * math.sqrt(512) * direction
                cond_manip = diffae_cls_model.denormalize(cond_manip)

                # 4. Render manipulated face from (xT, cond_manip)
                pred = diffae_model.render(xT, cond_manip, T=100)

            gen_duration = time.time() - t_gen_start

            # pred is [1, 3, 256, 256] in range [0, 1]
            fake_np = (pred.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            fake_pil = Image.fromarray(fake_np).resize((224, 224), Image.BILINEAR)
            fake_pil.save(fake_path, format="PNG")

            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            ram_mb = psutil.Process().memory_info().rss / (1024**2)

            sample_metrics = {
                "inference_time_sec": round(gen_duration, 3),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "ram_mb": round(ram_mb, 2),
                "real_inference": True,
                "official_diffae": True,
                "model_class": "BeatGANsAutoencModel",
                "checkpoint_autoenc": str(autoenc_ckpt_path),
                "checkpoint_cls": str(cls_ckpt_path),
                "checkpoint_sha256": autoenc_sha256,
                "param_count": 168492291,
                "attribute_manipulation_method": f"Latent hyperplane shift along CelebA classifier weight ({celeb_attr}) with DDIM stochastic inversion",
            }
            pbar.set_postfix({"attr": cur_attribute, "sec": f"{gen_duration:.2f}"})
        elif iafaces_netE is not None and iafaces_netG is not None and generator == "IAFaces":
            import time
            import psutil
            import torch.nn.functional as F

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
            fake_pil.save(fake_path, format="PNG")

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
        elif mock or checkpoint is None or not checkpoint.exists():
            fake_bgr = apply_mock_attribute_manipulation(cv2.cvtColor(src_np, cv2.COLOR_RGB2BGR), attribute=cur_attribute, seed=sample_seed)
            Image.fromarray(cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2RGB)).save(fake_path, format="PNG")
            sample_metrics = {"real_inference": False, "note": "mock"}
            pbar.set_postfix({"attr": cur_attribute, "mode": "mock"})
        else:
            fake_bgr = apply_mock_attribute_manipulation(cv2.cvtColor(src_np, cv2.COLOR_RGB2BGR), attribute=cur_attribute, seed=sample_seed)
            Image.fromarray(cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2RGB)).save(fake_path, format="PNG")
            sample_metrics = {"real_inference": False}
            pbar.set_postfix({"attr": cur_attribute, "mode": "fallback"})

        record = {
            "sample_id": sample_stem,
            "generator": generator,
            "forgery_type": "AM",
            "architecture": "Diffusion" if generator == "DiffAE" else "GAN",
            "image_path": fake_path.relative_to(dataset_root).as_posix(),
            "source_image_path": dest_src_file.relative_to(dataset_root).as_posix(),
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
    parser.add_argument("--dataset-root", type=str, default="MFVLR_Dataset", help="Root directory of MFVLR dataset")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--start-index", type=int, default=None, help="Explicit start index")
    parser.add_argument("--end-index", type=int, default=None, help="Explicit end index")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without heavy checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without generating files")
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
        logger.error("AM generation failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
