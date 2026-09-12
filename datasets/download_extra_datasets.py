"""Downloader and preprocessor for Celeb-DF-v2 extracted face image dataset."""

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


CELEB_DF_URL = "https://huggingface.co/datasets/Thien0103/DeepFake_Extracted_Face_Images/resolve/main/celeb_df_extracted.zip"


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
                        chunk = resp.read(2 * 1024 * 1024)
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
    print(f"Extracting {zip_path.name} -> {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
            zip_ref.extract(member, extract_to)
    print(f"[OK] Extracted {zip_path.name}.")


def build_celeb_df_manifests(data_dir: Path, max_samples: Optional[int] = None) -> None:
    """Scan extracted Celeb-DF folder and build train/val/test manifests."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images_root = data_dir / "images"
    all_images = [p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]

    samples: List[Dict] = []
    for img_path in all_images:
        rel_path = img_path.relative_to(data_dir).as_posix()
        path_lower = rel_path.lower()
        # In Celeb-DF: Real folders usually have 'real', 'original', or 'youtube'; Fake folders have 'fake' or 'synthesis'
        is_fake = not ("real" in path_lower or "original" in path_lower or "youtube" in path_lower)
        samples.append({
            "image_path": rel_path,
            "label": 1 if is_fake else 0,
            "is_fake": is_fake,
            "manipulation_type": "FS" if is_fake else "Real",
            "generator": "FaceSwap" if is_fake else None,
            "family": "gan" if is_fake else None,
        })

    random.seed(42)
    random.shuffle(samples)
    print(f"Total valid image samples found in Celeb-DF: {len(samples)}")

    if not samples:
        print("[Warning] No image files found to generate manifests.")
        return

    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    n_train = int(len(samples) * 0.8)
    n_val = int(len(samples) * 0.1)

    train_data = samples[:n_train]
    val_data = samples[n_train : n_train + n_val]
    test_data = samples[n_train + n_val :]

    for data, fname in [(train_data, "train_manifest.jsonl"), (val_data, "val_manifest.jsonl"), (test_data, "test_manifest.jsonl")]:
        filepath = data_dir / fname
        with open(filepath, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        print(f"Saved {len(data)} to: {fname}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Celeb-DF-v2 extracted face dataset.")
    parser.add_argument("--data-dir", type=str, default="data/Celeb-DF-v2", help="Directory to store dataset.")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading zip if already present.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples in manifests.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    downloads_dir = data_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    zip_path = downloads_dir / "celeb_df_extracted.zip"

    if not args.skip_download:
        if zip_path.exists() and zip_path.stat().st_size > 100 * 1024 * 1024:
            print(f"[Skip] {zip_path.name} already downloaded.")
        else:
            print(f"--- Downloading {zip_path.name} ---")
            download_file(CELEB_DF_URL, zip_path)

    extract_to = data_dir / "images"
    if not extract_to.exists() or len(list(extract_to.glob("*"))) == 0:
        extract_zip(zip_path, extract_to)
    else:
        print("[Skip] Already extracted.")

    print("\n--- Generating Manifests ---")
    build_celeb_df_manifests(data_dir, max_samples=args.max_samples)
    print(f"[Done] Celeb-DF-v2 prepared successfully at: {data_dir}")


if __name__ == "__main__":
    main()
