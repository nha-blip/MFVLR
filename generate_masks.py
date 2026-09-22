"""Ground-truth localization mask generator for MFVLR / GenFace reproduction dataset.

Rules:
- Output mask dimensions: 224 x 224 pixels.
- Stored as lossless PNG with uint8 values {0, 255}:
    * 0: pristine / real / background
    * 255: manipulated / forged region
- REAL: mask is all zeros.
- EFS: mask is all ones (255 in PNG).
- AM & FS: computed from corresponding source and fake pair:
    diff = abs(fake - source)
    gray = 0.299*R + 0.587*G + 0.114*B
    norm = gray / 255.0
    mask = (norm > threshold) * 255
- Strict error checking: source-fake pair must exist and have consistent geometry.
  Never silently drop error samples.
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GenerateMasks")

LUMINANCE_WEIGHTS = (0.299, 0.587, 0.114)


def compute_difference_mask(
    fake_img: Image.Image,
    source_img: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    threshold: float = 0.1,
    luminance_weights: Tuple[float, float, float] = LUMINANCE_WEIGHTS,
) -> np.ndarray:
    """Compute binary difference mask between fake and source face images.

    Args:
        fake_img: Manipulated face PIL Image.
        source_img: Original source face PIL Image.
        target_size: (width, height) target dimensions, default (224, 224).
        threshold: Binarization threshold in [0, 1], default 0.1.
        luminance_weights: Weights for RGB to grayscale conversion.

    Returns:
        np.ndarray of shape (H, W) with dtype uint8 and values in {0, 255}.
    """
    # Deterministic resize to target size
    fake_resized = fake_img.convert("RGB").resize(target_size, Image.BILINEAR)
    src_resized = source_img.convert("RGB").resize(target_size, Image.BILINEAR)

    fake_arr = np.array(fake_resized, dtype=np.float32)
    src_arr = np.array(src_resized, dtype=np.float32)

    # 1. Absolute pixel-wise RGB difference
    diff = np.abs(fake_arr - src_arr)

    # 2. Grayscale conversion via luminance weights
    w_r, w_g, w_b = luminance_weights
    gray = diff[..., 0] * w_r + diff[..., 1] * w_g + diff[..., 2] * w_b

    # 3. Divide by 255 to normalize into [0, 1]
    norm_gray = gray / 255.0

    # 4. Threshold at 0.1 and map to {0, 255}
    binary_mask = (norm_gray > threshold).astype(np.uint8) * 255
    return binary_mask


def generate_single_mask(
    forgery_type: str,
    fake_path: Optional[Union[str, Path]] = None,
    source_path: Optional[Union[str, Path]] = None,
    target_size: Tuple[int, int] = (224, 224),
    threshold: float = 0.1,
) -> np.ndarray:
    """Generate ground-truth mask for a single sample according to its forgery type.

    Args:
        forgery_type: One of 'REAL', 'EFS', 'AM', 'FS'.
        fake_path: Path to fake image (required for AM/FS).
        source_path: Path to source image (required for AM/FS).
        target_size: (width, height) default (224, 224).
        threshold: Threshold for difference mask, default 0.1.

    Returns:
        uint8 numpy array of shape (H, W) with values {0, 255}.
    """
    f_type = str(forgery_type).upper()

    if f_type in ("REAL", "0"):
        # Real face: all zeros
        return np.zeros(target_size[::-1], dtype=np.uint8)

    if f_type in ("EFS", "ENTIRE_SYNTHESIS", "ENTIRE_FACE_SYNTHESIS"):
        # Entire face synthesis: all 255 (all ones)
        return np.full(target_size[::-1], 255, dtype=np.uint8)

    if f_type in ("AM", "FS", "FACE_SWAP", "ATTRIBUTE_MANIPULATION"):
        if not fake_path or not os.path.exists(fake_path):
            raise FileNotFoundError(f"[{f_type}] Fake image not found at: {fake_path}")
        if not source_path or not os.path.exists(source_path):
            raise FileNotFoundError(f"[{f_type}] Source image not found at: {source_path}. AM/FS requires source-fake pair!")

        with Image.open(fake_path) as f_img, Image.open(source_path) as s_img:
            return compute_difference_mask(
                fake_img=f_img,
                source_img=s_img,
                target_size=target_size,
                threshold=threshold,
            )

    raise ValueError(f"Unsupported forgery_type: '{forgery_type}'. Expected REAL, EFS, AM, or FS.")


def save_mask_lossless(mask_arr: np.ndarray, output_path: Union[str, Path]) -> None:
    """Save mask as a lossless PNG with uint8 values in {0, 255}."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray(mask_arr, mode="L")
    mask_img.save(path, format="PNG", compress_level=6)


def process_metadata_csv(
    metadata_path: Union[str, Path],
    dataset_root: Union[str, Path],
    threshold: float = 0.1,
    target_size: Tuple[int, int] = (224, 224),
    resume: bool = True,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Process all samples defined in metadata CSV and generate ground-truth masks.

    Args:
        metadata_path: Path to metadata CSV file.
        dataset_root: Root directory of dataset to resolve relative paths.
        threshold: Threshold for difference mask.
        target_size: Target mask size (224, 224).
        resume: Skip existing masks if True.
        dry_run: Print actions without writing files.

    Returns:
        Dictionary of processing statistics.
    """
    root = Path(dataset_root)
    meta_file = Path(metadata_path)

    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    with open(meta_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    total = len(reader)
    logger.info(f"Loaded {total} records from {meta_file}")

    stats = {"total": total, "created": 0, "skipped": 0, "errors": 0}
    errors: List[str] = []

    for idx, row in enumerate(tqdm(reader, desc="Generating Masks")):
        sample_id = row.get("sample_id", f"sample_{idx:06d}")
        forgery_type = row.get("forgery_type", "REAL")
        mask_rel_path = row.get("mask_path")

        if not mask_rel_path:
            err = f"Sample {sample_id} is missing 'mask_path' field in metadata."
            errors.append(err)
            stats["errors"] += 1
            continue

        mask_full_path = root / mask_rel_path

        if resume and mask_full_path.exists() and mask_full_path.stat().st_size > 0:
            stats["skipped"] += 1
            continue

        # Resolve paths
        fake_rel = row.get("image_path")
        src_rel = row.get("source_image_path") or row.get("source_path")

        fake_full = (root / fake_rel) if fake_rel else None
        src_full = (root / src_rel) if src_rel else None

        if dry_run:
            logger.info(f"[Dry Run] Would generate mask: {mask_full_path} (Type: {forgery_type})")
            stats["created"] += 1
            continue

        try:
            mask = generate_single_mask(
                forgery_type=forgery_type,
                fake_path=fake_full,
                source_path=src_full,
                target_size=target_size,
                threshold=threshold,
            )
            save_mask_lossless(mask, mask_full_path)
            stats["created"] += 1
        except Exception as e:
            err_msg = f"Error processing sample {sample_id} ({fake_rel}): {e}"
            logger.error(err_msg)
            errors.append(err_msg)
            stats["errors"] += 1

    logger.info(f"Mask Generation Summary: {stats}")
    if errors:
        logger.error(f"Encountered {len(errors)} errors during mask generation!")
        raise RuntimeError(f"Mask generation failed with {len(errors)} errors:\n" + "\n".join(errors[:10]))

    return stats


class MaskGenerator:
    """Class interface for ground-truth mask generation."""

    def __init__(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        target_size: Tuple[int, int] = (224, 224),
        threshold: float = 0.1,
    ) -> None:
        self.dataset_root = Path(dataset_root) if dataset_root else Path(".")
        self.target_size = target_size
        self.threshold = threshold

    def generate_mask(
        self,
        label: int,
        forgery_type: str,
        fake_img: Union[np.ndarray, Image.Image],
        source_img: Optional[Union[np.ndarray, Image.Image]] = None,
    ) -> np.ndarray:
        """Generate mask directly from image array or PIL Image objects."""
        ftype = str(forgery_type).upper()
        if label == 0 or ftype in ("REAL", "0"):
            return np.zeros(self.target_size[::-1], dtype=np.uint8)

        if ftype in ("EFS", "ENTIRE_SYNTHESIS", "ENTIRE_FACE_SYNTHESIS"):
            return np.full(self.target_size[::-1], 255, dtype=np.uint8)

        if ftype in ("AM", "FS", "FACE_SWAP", "ATTRIBUTE_MANIPULATION"):
            if source_img is None:
                raise ValueError(f"Source image is required for {forgery_type} mask generation!")

            # Convert to PIL if numpy array
            if isinstance(fake_img, np.ndarray):
                f_pil = Image.fromarray(fake_img)
            else:
                f_pil = fake_img

            if isinstance(source_img, np.ndarray):
                s_pil = Image.fromarray(source_img)
            else:
                s_pil = source_img

            return compute_difference_mask(
                fake_img=f_pil,
                source_img=s_pil,
                target_size=self.target_size,
                threshold=self.threshold,
            )

        raise ValueError(f"Unsupported forgery_type: '{forgery_type}' with label {label}")

    def generate_from_metadata(
        self,
        metadata_csv: Union[str, Path],
        resume: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """Processes all samples from a metadata CSV file."""
        stats = process_metadata_csv(
            metadata_path=metadata_csv,
            dataset_root=self.dataset_root,
            threshold=self.threshold,
            target_size=self.target_size,
            resume=resume,
            dry_run=dry_run,
        )
        return {
            "processed": stats.get("created", 0) + stats.get("skipped", 0),
            "created": stats.get("created", 0),
            "skipped": stats.get("skipped", 0),
            "failed": stats.get("errors", 0),
        }


def main():
    parser = argparse.ArgumentParser(description="Generate ground-truth masks for MFVLR dataset.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV file")
    parser.add_argument("--root", type=str, default=".", help="Dataset root directory")
    parser.add_argument("--threshold", type=float, default=0.1, help="Difference threshold (default: 0.1)")
    parser.add_argument("--size", type=int, default=224, help="Target mask dimension (default: 224)")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing masks")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without writing mask files")
    args = parser.parse_args()

    process_metadata_csv(
        metadata_path=args.metadata,
        dataset_root=args.root,
        threshold=args.threshold,
        target_size=(args.size, args.size),
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

