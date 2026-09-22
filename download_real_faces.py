#!/usr/bin/env python3
"""
download_real_faces.py - High-Throughput Real Face Dataset Downloader & Normalizer

Downloads up to 50,000+ pristine real faces from the two primary GenFace benchmarks:
1. CelebA-HQ (korexyz/celeba-hq-256x256) - 30,000 celebrity faces
2. FFHQ (bitmind/ffhq-256) - 70,000 Flickr diverse human faces

Key Features:
- Automatically balances sources (e.g. 25,000 CelebA-HQ + 25,000 FFHQ = 50,000 total).
- Streaming download: fetches one parquet chunk at a time, extracts, and removes chunk to preserve disk space.
- Normalizes all images to 224x224 RGB PNG format (GenFace benchmark standard).
- Resume support: skips already downloaded and extracted faces.
- Saves provenance manifest (real_provenance.json) recording origin and index.
- Multi-threaded disk write for maximum I/O throughput.
"""

import argparse
import io
import json
import logging
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_real_faces")

CELEBA_HQ_URL_TEMPLATE = "https://huggingface.co/datasets/korexyz/celeba-hq-256x256/resolve/main/data/train-{shard:05d}-of-00006.parquet"
CELEBA_HQ_TOTAL_SHARDS = 6
CELEBA_HQ_PER_SHARD = 5000

FFHQ_URL_TEMPLATE = "https://huggingface.co/datasets/bitmind/ffhq-256/resolve/main/data/train-{shard:05d}-of-00016.parquet"
FFHQ_TOTAL_SHARDS = 16
FFHQ_PER_SHARD = 4375


def download_chunk(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Downloads a file over HTTP with a progress bar and retries."""
    import time
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d) ...", url, attempt, max_retries)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp, open(temp_path, "wb") as f:
                total_size = int(resp.headers.get("content-length", 0))
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
                    while True:
                        buffer = resp.read(1024 * 256)
                        if not buffer:
                            break
                        f.write(buffer)
                        pbar.update(len(buffer))

            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            return True
        except Exception as e:
            logger.warning("Download attempt %d/%d failed for %s: %s", attempt, max_retries, url, e)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            if attempt < max_retries:
                time.sleep(2 * attempt)

    logger.error("All %d download attempts failed for %s", max_retries, url)
    return False


def save_single_image(args_tuple):
    """Worker function to resize and save a single image."""
    img_bytes, out_path = args_tuple
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            rgb_img = img.convert("RGB")
            if rgb_img.size != (224, 224):
                rgb_img = rgb_img.resize((224, 224), Image.BILINEAR)
            rgb_img.save(out_path, format="PNG")
        return True
    except Exception as e:
        logger.error("Error saving image %s: %s", out_path.name, e)
        return False


def process_dataset_shards(
    source_name: str,
    url_template: str,
    total_shards: int,
    images_per_shard: int,
    target_count: int,
    out_dir: Path,
    start_global_idx: int,
    keep_parquet: bool = False,
    max_workers: int = 4,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Downloads shards sequentially and extracts images until target_count is reached."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas and pyarrow are required. Run: pip install pandas pyarrow")
        sys.exit(1)

    records: List[Dict[str, Any]] = []
    extracted_so_far = 0
    current_global_idx = start_global_idx
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for shard_idx in range(total_shards):
        if extracted_so_far >= target_count:
            break

        images_in_shard = min(images_per_shard, target_count - extracted_so_far)

        # Smart resume check: if all images for this shard already exist, skip downloading shard!
        all_shard_images_exist = True
        for i in range(images_in_shard):
            chk_file = out_dir / f"real_{current_global_idx + i:06d}.png"
            if not chk_file.exists() or chk_file.stat().st_size == 0:
                all_shard_images_exist = False
                break

        if all_shard_images_exist:
            logger.info("Shard %d of %s already fully extracted (%d images exist). Skipping shard download.", shard_idx, source_name, images_in_shard)
            for i in range(images_in_shard):
                rec_id = f"real_{current_global_idx + i:06d}"
                out_file = out_dir / f"{rec_id}.png"
                try:
                    rel_path = str(out_file.relative_to(out_dir.parent.parent if out_dir.parent.name == 'images' else out_dir).as_posix())
                except ValueError:
                    rel_path = str(out_file.name)

                records.append({
                    "sample_id": rec_id,
                    "original_dataset": source_name,
                    "source_shard": shard_idx,
                    "source_row": i,
                    "image_path": rel_path,
                    "resolution": "224x224",
                    "label": "real",
                    "forgery_type": "REAL",
                })
            extracted_so_far += images_in_shard
            current_global_idx += images_in_shard
            continue

        shard_url = url_template.format(shard=shard_idx)
        parquet_file = cache_dir / f"{source_name}_shard_{shard_idx:05d}.parquet"

        if dry_run:
            logger.info("[DRY-RUN] Would process %s shard %d from %s (%d images)", source_name, shard_idx, shard_url, images_in_shard)
            extracted_so_far += images_in_shard
            current_global_idx += images_in_shard
            continue

        if not parquet_file.exists():
            success = download_chunk(shard_url, parquet_file)
            if not success:
                logger.error("Skipping %s shard %d due to download failure.", source_name, shard_idx)
                continue

        logger.info("Reading %s shard %d ...", source_name, shard_idx)
        try:
            df = pd.read_parquet(str(parquet_file))
        except Exception as e:
            logger.error("Failed to read parquet %s: %s", parquet_file.name, e)
            continue

        tasks = []
        shard_records = []
        for row_idx, row in df.iterrows():
            if extracted_so_far >= target_count:
                break

            out_filename = f"real_{current_global_idx:06d}.png"
            out_file = out_dir / out_filename

            img_item = row["image"]
            if isinstance(img_item, dict):
                img_bytes = img_item.get("bytes")
            elif hasattr(img_item, "tobytes"):
                img_bytes = img_item.tobytes()
            else:
                img_bytes = img_item

            try:
                rel_path = str(out_file.relative_to(out_dir.parent.parent if out_dir.parent.name == 'images' else out_dir).as_posix())
            except ValueError:
                rel_path = str(out_file.name)

            shard_records.append({
                "sample_id": f"real_{current_global_idx:06d}",
                "original_dataset": source_name,
                "source_shard": shard_idx,
                "source_row": row_idx,
                "image_path": rel_path,
                "resolution": "224x224",
                "label": "real",
                "forgery_type": "REAL",
            })

            # Only task to thread pool if missing
            if not out_file.exists() or out_file.stat().st_size == 0:
                tasks.append((img_bytes, out_file))

            extracted_so_far += 1
            current_global_idx += 1

        if tasks:
            logger.info("Extracting %d new images from %s shard %d ...", len(tasks), source_name, shard_idx)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(executor.map(save_single_image, tasks), total=len(tasks), desc=f"Saving {source_name}"))
        else:
            logger.info("All images in %s shard %d were already extracted.", source_name, shard_idx)

        records.extend(shard_records)

        # Remove downloaded parquet to save disk space if requested
        if not keep_parquet and parquet_file.exists():
            try:
                parquet_file.unlink()
            except OSError:
                pass

    if not keep_parquet and cache_dir.exists():
        try:
            if not any(cache_dir.iterdir()):
                cache_dir.rmdir()
        except OSError:
            pass

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract 50,000 pristine real face images from CelebA-HQ and FFHQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=50000, help="Total number of real face images to acquire")
    parser.add_argument("--out-dir", type=str, default="MFVLR_Dataset/images/real", help="Destination folder for real images")
    parser.add_argument("--celeba-ratio", type=float, default=0.5, help="Proportion of images from CelebA-HQ (0.0 to 1.0)")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel threads for image resizing/saving")
    parser.add_argument("--keep-parquet", action="store_true", help="Keep downloaded parquet cache files")
    parser.add_argument("--dry-run", action="store_true", help="Print download plan without writing files")

    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    celeba_count = int(args.count * args.celeba_ratio)
    ffhq_count = args.count - celeba_count

    logger.info("==========================================================")
    logger.info("  MFVLR REAL FACE DATASET ACQUISITION PLAN")
    logger.info("  Total Target Images: %d", args.count)
    logger.info("  - Source 1 (CelebA-HQ): %d images", celeba_count)
    logger.info("  - Source 2 (FFHQ):      %d images", ffhq_count)
    logger.info("  Destination Directory:  %s", out_dir)
    logger.info("==========================================================")

    # 1. Process CelebA-HQ
    logger.info("--- Phase 1: Acquiring %d faces from CelebA-HQ ---", celeba_count)
    celeba_records = process_dataset_shards(
        source_name="CelebA-HQ",
        url_template=CELEBA_HQ_URL_TEMPLATE,
        total_shards=CELEBA_HQ_TOTAL_SHARDS,
        images_per_shard=CELEBA_HQ_PER_SHARD,
        target_count=celeba_count,
        out_dir=out_dir,
        start_global_idx=0,
        keep_parquet=args.keep_parquet,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )

    # 2. Process FFHQ
    logger.info("--- Phase 2: Acquiring %d faces from FFHQ ---", ffhq_count)
    ffhq_records = process_dataset_shards(
        source_name="FFHQ",
        url_template=FFHQ_URL_TEMPLATE,
        total_shards=FFHQ_TOTAL_SHARDS,
        images_per_shard=FFHQ_PER_SHARD,
        target_count=ffhq_count,
        out_dir=out_dir,
        start_global_idx=len(celeba_records),
        keep_parquet=args.keep_parquet,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )

    total_records = celeba_records + ffhq_records
    logger.info("Acquisition complete: %d real faces saved to %s", len(total_records), out_dir)

    # 3. Save provenance manifest
    if not args.dry_run and total_records:
        manifest_path = out_dir / "real_provenance.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    old_records = json.load(f)
                merged = {r["sample_id"]: r for r in old_records if "sample_id" in r}
                for r in total_records:
                    merged[r["sample_id"]] = r
                total_records = list(merged.values())
            except Exception:
                pass

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(total_records, f, indent=2)
        logger.info("Saved provenance manifest with %d items to %s", len(total_records), manifest_path)


if __name__ == "__main__":
    main()
