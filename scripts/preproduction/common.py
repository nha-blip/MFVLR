import os
import sys
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

# Fix Windows stdout encoding for non-ASCII paths
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PREPROD_ROOT = Path(__file__).resolve().parent.parent.parent / "GenFace_Reproduced" / "preproduction_500"

METADATA_FIELDS = [
    "sample_id",
    "image_path",
    "source_path",
    "target_path",
    "mask_path",
    "label",
    "forgery_type",
    "architecture",
    "generator",
    "split",
    "L1",
    "L2",
    "L3",
    "L4",
    "original_dataset",
    "generator_repo",
    "generator_commit",
    "checkpoint",
    "checkpoint_sha256",
    "seed",
    "source_id",
    "target_id",
    "generation_config",
    "native_resolution",
    "sha256",
    "notes",
]

LUMINANCE_WEIGHTS = (0.299, 0.587, 0.114)


def compute_sha256(filepath: Union[str, Path]) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def compute_difference_mask(
    fake_img: Image.Image,
    ref_img: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    threshold: float = 0.1,
    luminance_weights: Tuple[float, float, float] = LUMINANCE_WEIGHTS,
) -> np.ndarray:
    """Strict MFVLR / GenFace mask protocol:
    1. abs(fake - reference) on RGB
    2. ITU-R BT.601 luminance: gray = 0.299*R + 0.587*G + 0.114*B
    3. gray / 255.0
    4. threshold > 0.1
    5. binary uint8 {0, 255}
    6. NO morphology
    """
    fake_resized = fake_img.convert("RGB").resize(target_size, Image.BILINEAR)
    ref_resized = ref_img.convert("RGB").resize(target_size, Image.BILINEAR)

    fake_arr = np.array(fake_resized, dtype=np.float32)
    ref_arr = np.array(ref_resized, dtype=np.float32)

    diff = np.abs(fake_arr - ref_arr)
    w_r, w_g, w_b = luminance_weights
    gray = diff[..., 0] * w_r + diff[..., 1] * w_g + diff[..., 2] * w_b
    norm_gray = gray / 255.0
    binary_mask = (norm_gray > threshold).astype(np.uint8) * 255
    return binary_mask


def save_mask_lossless(mask_arr: np.ndarray, output_path: Union[str, Path]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray(mask_arr, mode="L")
    mask_img.save(path, format="PNG", compress_level=6)


def init_workspace() -> None:
    dirs = [
        PREPROD_ROOT / "images" / "EFS",
        PREPROD_ROOT / "images" / "AM",
        PREPROD_ROOT / "images" / "FS",
        PREPROD_ROOT / "source" / "AM",
        PREPROD_ROOT / "source" / "FS",
        PREPROD_ROOT / "target" / "FS",
        PREPROD_ROOT / "masks" / "EFS",
        PREPROD_ROOT / "masks" / "AM",
        PREPROD_ROOT / "masks" / "FS",
        PREPROD_ROOT / "metadata",
        PREPROD_ROOT / "logs",
        PREPROD_ROOT / "verification",
        PREPROD_ROOT / "visual_grids",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_existing_records(csv_path: Union[str, Path]) -> Dict[str, Dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        return {}
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records[row["sample_id"]] = row
    return records


def is_sample_valid_and_complete(
    sample_id: str,
    record: Dict[str, str],
    required_files: List[Path],
) -> bool:
    """Verifies sample is complete and valid before allowing skip during resume."""
    if not record or record.get("sample_id") != sample_id:
        return False

    # Check all required files exist, readable, non-empty
    for p in required_files:
        if not p.exists() or p.stat().st_size == 0:
            return False
        try:
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:
                if img.size != (224, 224):
                    return False
        except Exception:
            return False

    # Check mask strictly binary {0, 255}
    mask_path = PREPROD_ROOT / record["mask_path"]
    try:
        with Image.open(mask_path) as m_img:
            m_arr = np.array(m_img)
            unique_vals = set(np.unique(m_arr))
            if not unique_vals.issubset({0, 255}):
                return False
    except Exception:
        return False

    return True


def append_metadata_record(csv_path: Union[str, Path], record: Dict[str, str]) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def create_contact_sheet(
    image_rows: List[List[Image.Image]],
    output_path: Union[str, Path],
    cell_size: Tuple[int, int] = (224, 224),
) -> None:
    if not image_rows or not image_rows[0]:
        return
    num_rows = len(image_rows)
    num_cols = len(image_rows[0])
    sheet = Image.new("RGB", (num_cols * cell_size[0], num_rows * cell_size[1]))

    for r_idx, row in enumerate(image_rows):
        for c_idx, cell_img in enumerate(row):
            img_rgb = cell_img.convert("RGB").resize(cell_size)
            sheet.paste(img_rgb, (c_idx * cell_size[0], r_idx * cell_size[1]))

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_p, format="PNG")
