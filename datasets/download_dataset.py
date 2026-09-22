"""Automated downloader and preprocessor for OpenRL/DeepFakeFace dataset.

Downloads zip files from HuggingFace, extracts them, and generates
train_manifest.jsonl, val_manifest.jsonl, and test_manifest.jsonl for MFVLR.
"""

import argparse
import json
import os
import random
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Fix Windows cp1252 UnicodeEncodeError for Vietnamese folder names
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


BASE_URL = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/"
DATASET_FILES = {
    "wiki.zip": {
        "url": BASE_URL + "wiki.zip",
        "category": "wiki",
        "is_fake": False,
        "manipulation_type": "Real",
        "generator": None,
        "family": None,
    },
    "text2img.zip": {
        "url": BASE_URL + "text2img.zip",
        "category": "text2img",
        "is_fake": True,
        "manipulation_type": "EFS",
        "generator": "DDPM",
        "family": "diffusion",
    },
    "inpainting.zip": {
        "url": BASE_URL + "inpainting.zip",
        "category": "inpainting",
        "is_fake": True,
        "manipulation_type": "AM",
        "generator": "DiffAE",
        "family": "diffusion",
    },
    "insight.zip": {
        "url": BASE_URL + "insight.zip",
        "category": "insight",
        "is_fake": True,
        "manipulation_type": "FS",
        "generator": "FaceSwapper",
        "family": "diffusion",
    },
}


def download_file(url: str, dest_path: Path, max_retries: int = 5) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    headers = {"User-Agent": "Mozilla/5.0"}
    downloaded_bytes = 0

    if temp_path.exists():
        downloaded_bytes = temp_path.stat().st_size

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            if downloaded_bytes > 0:
                req.add_header("Range", f"bytes={downloaded_bytes}-")

            with urllib.request.urlopen(req, timeout=30) as resp:
                total_size = int(resp.headers.get("Content-Length", 0))
                if downloaded_bytes > 0 and resp.status == 206:
                    total_size += downloaded_bytes

                mode = "ab" if downloaded_bytes > 0 else "wb"
                desc = f"Downloading {dest_path.name}"
                with open(temp_path, mode) as f, tqdm(
                    total=total_size,
                    initial=downloaded_bytes,
                    unit="B",
                    unit_scale=True,
                    desc=desc,
                ) as pbar:
                    while True:
                        chunk = resp.read(1024 * 1024)  # 1MB chunk
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

            temp_path.rename(dest_path)
            print(f"[OK] Downloaded {dest_path.name} successfully.")
            return
        except Exception as e:
            print(f"[Warning] Attempt {attempt}/{max_retries} failed for {url}: {e}")
            if temp_path.exists():
                downloaded_bytes = temp_path.stat().st_size
            if attempt == max_retries:
                raise e


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f"Extracting {zip_path.name}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
            zip_ref.extract(member, extract_to)
    print(f"[OK] Extracted {zip_path.name}.")


def build_manifests(
    data_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_per_category: Optional[int] = None,
    categories: Optional[List[str]] = None,
) -> None:
    random.seed(seed)
    images_dir = data_dir / "images"
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    all_samples: List[Dict] = []

    target_files = {k: v for k, v in DATASET_FILES.items() if not categories or v['category'] in categories or k in categories}
    for filename, meta in target_files.items():
        cat = meta["category"]
        cat_dir = images_dir / cat
        if not cat_dir.exists():
            candidates = list(images_dir.glob(f"**/{cat}"))
            if candidates:
                cat_dir = candidates[0]
            else:
                print(f"[Warning] Category directory {cat} not found. Skipping {cat}.")
                continue

        # Find all images recursively
        cat_images = [
            p for p in cat_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_exts
        ]

        if max_per_category and len(cat_images) > max_per_category:
            random.shuffle(cat_images)
            cat_images = cat_images[:max_per_category]

        print(f"Found {len(cat_images)} images for category '{cat}'")

        for img_path in cat_images:
            rel_path = img_path.relative_to(data_dir).as_posix()
            record = {
                "image_path": rel_path,
                "label": 1 if meta["is_fake"] else 0,
                "is_fake": meta["is_fake"],
                "manipulation_type": meta["manipulation_type"],
            }
            if meta["generator"]:
                record["generator"] = meta["generator"]
            if meta["family"]:
                record["family"] = meta["family"]
            all_samples.append(record)

    random.shuffle(all_samples)
    total_count = len(all_samples)
    print(f"Total collected samples across categories: {total_count}")

    if total_count == 0:
        print("[Error] No samples found to generate manifests!")
        return

    n_train = int(total_count * train_ratio)
    n_val = int(total_count * val_ratio)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train : n_train + n_val]
    test_samples = all_samples[n_train + n_val :]

    def write_jsonl(samples: List[Dict], filepath: Path):
        with open(filepath, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"Saved {len(samples)} samples to: {filepath.name}")

    write_jsonl(train_samples, data_dir / "train_manifest.jsonl")
    write_jsonl(val_samples, data_dir / "val_manifest.jsonl")
    write_jsonl(test_samples, data_dir / "test_manifest.jsonl")


def update_config(config_path: Path, data_dir: Path) -> None:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("dataset", {})
    cfg["dataset"]["root"] = data_dir.as_posix()
    cfg["dataset"]["train_manifest"] = (data_dir / "train_manifest.jsonl").as_posix()
    cfg["dataset"]["val_manifest"] = (data_dir / "val_manifest.jsonl").as_posix()
    cfg["dataset"]["test_manifest"] = (data_dir / "test_manifest.jsonl").as_posix()

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[OK] Updated {config_path.name} with new dataset paths.")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare DeepFakeFace dataset.")
    parser.add_argument("--data-dir", type=str, default="data/DeepFakeFace", help="Directory to store dataset.")
    parser.add_argument("--categories", nargs="+", default=None, help="Categories to process (e.g. wiki, text2img, inpainting, insight). Default: all.")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading zips if already present.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extracting zips.")
    parser.add_argument("--max-per-category", type=int, default=None, help="Limit images per category for smaller subset.")
    parser.add_argument("--update-config", action="store_true", default=True, help="Update configs/mfvlr.yaml.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    downloads_dir = data_dir / "downloads"
    images_dir = data_dir / "images"

    downloads_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download each zip file first
    selected_files = {k: v for k, v in DATASET_FILES.items() if not args.categories or v['category'] in args.categories or k in args.categories}
    for filename, meta in selected_files.items():
        zip_path = downloads_dir / filename
        if not args.skip_download:
            if zip_path.exists() and zip_path.stat().st_size > 100 * 1024 * 1024:
                print(f"[Skip] {filename} already downloaded.")
            else:
                print(f"\n--- Downloading {filename} ---")
                download_file(meta["url"], zip_path)

    # 2. Extract each zip file
    if not args.skip_extract:
        for filename, meta in selected_files.items():
            zip_path = downloads_dir / filename
            extract_target = images_dir / meta["category"]
            if not extract_target.exists() or len(list(extract_target.glob("*"))) == 0:
                extract_zip(zip_path, extract_target)
            else:
                print(f"[Skip] {meta['category']} already extracted.")

    # 3. Generate manifests
    print("\n--- Generating Manifests ---")
    # Ensure images/real directory or symlink exists for AM source images
    wiki_dir = images_dir / "wiki"
    real_dir = images_dir / "real"
    if wiki_dir.exists() and not real_dir.exists():
        try:
            real_dir.symlink_to(wiki_dir, target_is_directory=True)
            print("[OK] Symlinked images/real -> images/wiki for AM generators")
        except Exception:
            try:
                import shutil
                shutil.copytree(wiki_dir, real_dir, dirs_exist_ok=True)
                print("[OK] Copied images/wiki -> images/real")
            except Exception as e:
                print(f"[Warning] Could not link real directory: {e}")

    build_manifests(data_dir, max_per_category=args.max_per_category, categories=args.categories)

    # 4. Update config
    if args.update_config:
        config_path = Path("configs/mfvlr.yaml").resolve()
        if config_path.exists():
            update_config(config_path, data_dir)


if __name__ == "__main__":
    main()
