"""Download and prepare the DiFF (Diffusion Facial Forgery) dataset for cross-dataset evaluation."""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set

import gdown
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REAL_FILE_ID = "1I6OnfSGlH7GxfYzRSxYJVUQn_kBTxs06"   # DiFF_real.zip (854 MB)
PMT_FILE_ID = "17YCwQxZknzkKiOMHIGH7t6JssHUSe6Fd"    # DiFF_pmt.txt (1.78 MB)

# 13 methods categorized into 4 conditions
TEST_FAKE_FILES = {
    "FE": {
        "CoDiff": {"id": "1eS6jqYef2rpJeGj09eTZO1himhtZGwjk", "size_mb": 262.2, "zip_name": "CoDiff.zip"},
        "cycle_diff": {"id": "1ybCEnyvVg9rOzP4jpgGpciinjhj1yvWV", "size_mb": 1800.0, "zip_name": "cycle_diff.zip"},
        "Imagic": {"id": "1KW0Rc7s15wuVjsPVDJMkS8n5Utnsr1rq", "size_mb": 1290.0, "zip_name": "Imagic.zip"},
    },
    "FS": {
        "DCFace": {"id": "179B3BshPPCTlMDPjGfuYG5CmInyj5k_Q", "size_mb": 20.0, "zip_name": "DCFace.zip"},
        "DiffFace": {"id": "1jFoBGd9O2LxdAN-uM-XRYgkL9lAgcdua", "size_mb": 104.4, "zip_name": "DiffFace.zip"},
    },
    "I2I": {
        "FreeDoM_I": {"id": "1fxFllmolDF_V5vssGoWCl3P0YAOrLF1D", "size_mb": 144.6, "zip_name": "FreeDoM_I.zip"},
        "LoRA": {"id": "10m1NbbHa25ZWf9I7K_rOkyKa_M9V1_SF", "size_mb": 888.6, "zip_name": "LoRA.zip"},
        "DreamBooth": {"id": "13g5clbzZ3b7eVbp-_09a6n4Xg9COBxBZ", "size_mb": 1400.0, "zip_name": "DreamBooth.zip"},
        "SDXL_Refine": {"id": "1KWdh2xFpsa5XQa2x6wakGTffGSfIJl4Z", "size_mb": 2050.0, "zip_name": "SDXL_Refine.zip"},
    },
    "T2I": {
        "FreeDoM_T": {"id": "18K3LuDl_UJmBLTWJFajNUkGbOVVXlBpU", "size_mb": 146.3, "zip_name": "FreeDoM_T.zip"},
        "HPS": {"id": "1f6XCD29oNvlngf8eeAkCxu4q-9D66fMd", "size_mb": 1420.0, "zip_name": "HPS.zip"},
        "Midjourney": {"id": "1Tjfiqv-Xi-le2rdOME_tem0cHU0iHNHV", "size_mb": 2140.0, "zip_name": "Midjourney.zip"},
        "SDXL": {"id": "1zacbwDwm5X-DNKqJrv6CamVE4hWO109j", "size_mb": 2210.0, "zip_name": "SDXL.zip"},
    },
}

# Representative balanced subset covering all 4 conditions (~3.6 GB total download)
REPRESENTATIVE_METHODS = {
    "FS": ["DCFace", "DiffFace"],
    "FE": ["CoDiff"],
    "I2I": ["FreeDoM_I", "LoRA"],
    "T2I": ["FreeDoM_T", "HPS"],
}


def download_drive_file(file_id: str, dest_path: Path) -> Path:
    """Download file from Google Drive using gdown with resume and virus warning bypass."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[Skip] {dest_path.name} already exists ({dest_path.stat().st_size / (1024**2):.1f} MB).")
        return dest_path

    print(f"\n--- Downloading {dest_path.name} (ID: {file_id}) ---")
    out = gdown.download(id=file_id, output=str(dest_path), quiet=False, resume=True)
    if not out or not dest_path.exists():
        raise RuntimeError(f"Failed to download file ID {file_id} to {dest_path}")
    print(f"[OK] Downloaded {dest_path.name} ({dest_path.stat().st_size / (1024**2):.1f} MB).")
    return dest_path


def extract_zip(zip_path: Path, extract_to: Path, delete_zip: bool = True) -> None:
    """Extract zip archive with progress bar and optionally delete zip to save disk space."""
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
            z.extract(member, extract_to)
    print(f"[OK] Extracted {zip_path.name}.")

    if delete_zip:
        try:
            zip_path.unlink()
            print(f"[Clean] Removed {zip_path.name} to free disk space.")
        except Exception as e:
            print(f"[Warning] Could not remove {zip_path.name}: {e}")


def prepare_diff_dataset(
    data_dir: Path,
    subset_mode: str = "representative",
    custom_methods: Optional[List[str]] = None,
    skip_download: bool = False,
    skip_extract: bool = False,
    keep_zips: bool = False,
    max_samples_per_method: Optional[int] = None,
) -> None:
    """Download, extract, and build manifests for DiFF dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = data_dir / "archives"
    images_dir = data_dir / "images"

    # Step 1: Real Images
    real_extract_dir = images_dir / "Real"
    if not skip_download and not (real_extract_dir.exists() and any(real_extract_dir.iterdir())):
        real_zip = archives_dir / "DiFF_real.zip"
        download_drive_file(REAL_FILE_ID, real_zip)
        if not skip_extract:
            extract_zip(real_zip, real_extract_dir, delete_zip=not keep_zips)
    else:
        print("[Info] DiFF Real images already present or download skipped.")

    # Step 2: Download Prompt Text
    pmt_path = data_dir / "DiFF_pmt.txt"
    if not skip_download and not pmt_path.exists():
        download_drive_file(PMT_FILE_ID, pmt_path)

    # Step 3: Determine which Fake methods to download
    methods_to_process: List[tuple[str, str, Dict]] = []
    for cat, methods in TEST_FAKE_FILES.items():
        for m_name, m_info in methods.items():
            if custom_methods:
                if m_name.lower() in [m.lower() for m in custom_methods]:
                    methods_to_process.append((cat, m_name, m_info))
            elif subset_mode == "all":
                methods_to_process.append((cat, m_name, m_info))
            elif subset_mode == "representative":
                if m_name in REPRESENTATIVE_METHODS.get(cat, []):
                    methods_to_process.append((cat, m_name, m_info))

    print(f"\nProcessing {len(methods_to_process)} Fake methods: {[m[1] for m in methods_to_process]}")

    # Step 4: Download and extract each Fake method sequentially
    for cat, m_name, m_info in methods_to_process:
        fake_extract_dir = images_dir / "Fake" / cat / m_name
        if fake_extract_dir.exists() and any(fake_extract_dir.iterdir()):
            print(f"[Skip] {cat}/{m_name} already extracted.")
            continue

        if not skip_download:
            m_zip = archives_dir / m_info["zip_name"]
            download_drive_file(m_info["id"], m_zip)
            if not skip_extract:
                extract_zip(m_zip, fake_extract_dir, delete_zip=not keep_zips)

    # Step 5: Build Manifests
    print("\n--- Generating DiFF Manifests ---")
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    # Collect Real test images specifically
    real_test_dir = real_extract_dir / "DiFF_real" / "test"
    if not real_test_dir.exists():
        real_test_dir = real_extract_dir

    real_samples: List[Dict] = []
    if real_test_dir.exists():
        for p in real_test_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_exts:
                rel_path = p.relative_to(data_dir).as_posix()
                real_samples.append({
                    "image_path": rel_path,
                    "label": 0,
                    "is_fake": False,
                    "manipulation_type": "Real",
                    "generator": "Real",
                    "category": "Real",
                })
    print(f"Found {len(real_samples)} Real test images.")

    # Collect Fake images per category and method
    fake_samples: List[Dict] = []
    category_samples: Dict[str, List[Dict]] = {"FE": [], "FS": [], "I2I": [], "T2I": []}

    for cat in ["FE", "FS", "I2I", "T2I"]:
        cat_dir = images_dir / "Fake" / cat
        if not cat_dir.exists():
            continue

        for method_dir in cat_dir.iterdir():
            if not method_dir.is_dir():
                continue
            m_name = method_dir.name
            method_imgs = [
                p for p in method_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in valid_exts
                and not p.name.startswith("._") and "__MACOSX" not in p.parts
            ]
            if max_samples_per_method and len(method_imgs) > max_samples_per_method:
                import random
                random.seed(42)
                method_imgs = random.sample(method_imgs, max_samples_per_method)

            for p in method_imgs:
                rel_path = p.relative_to(data_dir).as_posix()
                sample = {
                    "image_path": rel_path,
                    "label": 1,
                    "is_fake": True,
                    "manipulation_type": cat,
                    "generator": m_name,
                    "category": cat,
                }
                fake_samples.append(sample)
                category_samples[cat].append(sample)
            print(f"  - Fake {cat}/{m_name}: {len(method_imgs)} images")

    print(f"Total Fake images collected: {len(fake_samples)}")

    import random
    random.seed(42)

    # Save overall full test manifest
    all_test_samples = real_samples + fake_samples
    overall_manifest_path = data_dir / "diff_test_manifest.jsonl"
    with open(overall_manifest_path, "w", encoding="utf-8") as f:
        for item in all_test_samples:
            f.write(json.dumps(item) + "\n")
    print(f"[Manifest Saved] {overall_manifest_path} ({len(all_test_samples)} samples: {len(real_samples)} Real, {len(fake_samples)} Fake)")

    # Save 1:1 balanced overall test manifest
    n_fake = len(fake_samples)
    sampled_real = random.sample(real_samples, min(n_fake, len(real_samples)))
    balanced_samples = sampled_real + fake_samples
    random.shuffle(balanced_samples)
    balanced_manifest_path = data_dir / "diff_test_balanced.jsonl"
    with open(balanced_manifest_path, "w", encoding="utf-8") as f:
        for item in balanced_samples:
            f.write(json.dumps(item) + "\n")
    print(f"[Manifest Saved] {balanced_manifest_path} (1:1 balanced, {len(balanced_samples)} samples: {len(sampled_real)} Real, {len(fake_samples)} Fake)")

    # Save per-condition 1:1 balanced manifests
    for cat, fakes in category_samples.items():
        if not fakes:
            continue
        n_cat_fake = len(fakes)
        cat_real = random.sample(real_samples, min(n_cat_fake, len(real_samples)))
        cat_samples = cat_real + fakes
        random.shuffle(cat_samples)
        cat_manifest_path = data_dir / f"diff_test_{cat}.jsonl"
        with open(cat_manifest_path, "w", encoding="utf-8") as f:
            for item in cat_samples:
                f.write(json.dumps(item) + "\n")
        print(f"[Manifest Saved] {cat_manifest_path} (1:1 balanced, {len(cat_samples)} samples: {len(cat_real)} Real, {n_cat_fake} Fake)")

    # Print summary statistics
    summary_path = data_dir / "dataset_summary.json"
    summary = {
        "total_samples": len(all_test_samples),
        "real_count": len(real_samples),
        "fake_count": len(fake_samples),
        "conditions": {
            cat: len(fakes) for cat, fakes in category_samples.items()
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary Saved] {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DiFF dataset for MFVLR cross-dataset evaluation.")
    parser.add_argument("--data-dir", type=str, default="data/DiFF", help="Directory to store DiFF dataset.")
    parser.add_argument("--subset", type=str, default="representative", choices=["representative", "all", "custom"], help="Subset to download.")
    parser.add_argument("--methods", type=str, default="", help="Comma-separated custom method names (if --subset custom).")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading files.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extracting archives.")
    parser.add_argument("--keep-zips", action="store_true", help="Keep zip files after extraction.")
    parser.add_argument("--max-samples-per-method", type=int, default=None, help="Max fake samples per method.")
    args = parser.parse_args()

    custom_m = [m.strip() for m in args.methods.split(",") if m.strip()] if args.subset == "custom" else None
    prepare_diff_dataset(
        data_dir=Path(args.data_dir),
        subset_mode=args.subset,
        custom_methods=custom_m,
        skip_download=args.skip_download,
        skip_extract=args.skip_extract,
        keep_zips=args.keep_zips,
        max_samples_per_method=args.max_samples_per_method,
    )
