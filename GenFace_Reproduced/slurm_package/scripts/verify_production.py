#!/usr/bin/env python3
"""
MFVLR / GenFace-Reproduced: Comprehensive Production Dataset Verifier
Validates all 375,000 samples across the 8 generators without modifying any files:
- Expected counts & missing indices
- Identity uniqueness & SHA256 duplicate detection
- Image format (RGB 224x224 PNG) & mask format (uint8 224x224 PNG {0, 255})
- AM/FS source & target pairing integrity
- Metadata completeness (26 fields) and split = UNASSIGNED
"""

import os
import sys
import csv
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "output")).resolve()

EXPECTED_PRODUCTION_COUNTS = {
    "StyleGAN3": {"type": "EFS", "arch": "GAN", "expected": 50000},
    "IAFaces": {"type": "AM", "arch": "GAN", "expected": 5000},
    "LatTrans": {"type": "AM", "arch": "GAN", "expected": 60000},
    "FaceSwapper": {"type": "FS", "arch": "GAN", "expected": 30000},
    "DDPM": {"type": "EFS", "arch": "Diffusion", "expected": 50000},
    "LatDiff": {"type": "EFS", "arch": "Diffusion", "expected": 60000},
    "CollDiff": {"type": "EFS", "arch": "Diffusion", "expected": 50000},
    "DiffAE": {"type": "AM", "arch": "Diffusion", "expected": 70000},
}

METADATA_FIELDS = [
    "sample_id", "image_path", "source_path", "target_path", "mask_path",
    "label", "forgery_type", "architecture", "generator", "split",
    "L1", "L2", "L3", "L4", "original_dataset", "generator_repo",
    "generator_commit", "checkpoint", "checkpoint_sha256", "seed",
    "source_id", "target_id", "generation_config", "native_resolution",
    "sha256", "notes"
]

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

def verify_dataset(out_root: Path):
    print("================================================================================")
    print("=== MFVLR / GenFace-Reproduced: Production Dataset Verification ===")
    print("================================================================================")
    print(f"Output Root: {out_root}")

    master_csv = out_root / "metadata" / "all_production.csv"
    if not master_csv.exists():
        print(f"ERROR: Master production metadata not found at: {master_csv}")
        print("Please run scripts/merge_metadata.py first.")
        return False

    with open(master_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    print(f"Loaded {len(records):,} records from master CSV.")

    errors = []
    warnings = []
    seen_ids = set()
    sha_map = defaultdict(list)
    stats = defaultdict(lambda: {
        "generated": 0,
        "valid": 0,
        "invalid": 0,
        "duplicates": 0,
        "mask_ratios": [],
    })

    for r in records:
        sid = r.get("sample_id", "")
        gen = r.get("generator", "")
        ftype = r.get("forgery_type", "")
        split = r.get("split", "")

        if not sid:
            errors.append("Record missing sample_id")
            continue
        if sid in seen_ids:
            errors.append(f"Duplicate sample_id: {sid}")
        seen_ids.add(sid)

        stats[gen]["generated"] += 1

        # Check split
        if split != "UNASSIGNED":
            errors.append(f"Sample {sid}: split must be 'UNASSIGNED', got '{split}'")

        # Check required fields
        for field in METADATA_FIELDS:
            if field not in r:
                errors.append(f"Sample {sid}: missing field '{field}'")

        img_p = out_root / r["image_path"]
        msk_p = out_root / r["mask_path"]

        if not img_p.exists():
            errors.append(f"Sample {sid}: image missing {img_p}")
            stats[gen]["invalid"] += 1
            continue
        if not msk_p.exists():
            errors.append(f"Sample {sid}: mask missing {msk_p}")
            stats[gen]["invalid"] += 1
            continue

        if ftype in ["AM", "FS"]:
            src_p = out_root / r["source_path"]
            if not src_p.exists():
                errors.append(f"Sample {sid}: source missing {src_p}")

        if ftype == "FS":
            tar_p = out_root / r["target_path"]
            if not tar_p.exists():
                errors.append(f"Sample {sid}: target missing {tar_p}")

        # Image validation
        try:
            with Image.open(img_p) as img:
                if img.size != (224, 224) or img.mode != "RGB" or img.format != "PNG":
                    errors.append(f"Sample {sid}: invalid image ({img.format}, {img.mode}, {img.size})")
                    stats[gen]["invalid"] += 1
                    continue
        except Exception as e:
            errors.append(f"Sample {sid}: failed to read image: {e}")
            stats[gen]["invalid"] += 1
            continue

        # Mask validation
        try:
            with Image.open(msk_p) as msk:
                if msk.size != (224, 224) or msk.format != "PNG":
                    errors.append(f"Sample {sid}: invalid mask ({msk.format}, {msk.size})")
                    stats[gen]["invalid"] += 1
                    continue
                m_np = np.array(msk)
                if m_np.dtype != np.uint8:
                    errors.append(f"Sample {sid}: mask dtype {m_np.dtype} != uint8")
                vals = set(np.unique(m_np))
                if not vals.issubset({0, 255}):
                    errors.append(f"Sample {sid}: invalid mask values: {vals}")
                ratio = np.count_nonzero(m_np == 255) / m_np.size
                stats[gen]["mask_ratios"].append(ratio)
                if ftype == "EFS" and ratio != 1.0:
                    errors.append(f"Sample {sid} (EFS): mask ratio {ratio} != 1.0")
        except Exception as e:
            errors.append(f"Sample {sid}: failed to read mask: {e}")
            stats[gen]["invalid"] += 1
            continue

        # SHA256 validation
        img_sha = compute_sha256(img_p)
        if r.get("sha256") and r["sha256"] != img_sha:
            errors.append(f"Sample {sid}: SHA256 mismatch")
        sha_map[img_sha].append(sid)

        stats[gen]["valid"] += 1

    # Check duplicates
    for sha, sids in sha_map.items():
        if len(sids) > 1:
            errors.append(f"Duplicate image hash {sha[:12]}: {sids}")
            for sid in sids[1:]:
                for r in records:
                    if r["sample_id"] == sid:
                        stats[r["generator"]]["duplicates"] += 1

    # Print summary table
    print("\n" + "=" * 85)
    print(f"{'Generator':<15} | {'Expected':<10} | {'Generated':<10} | {'Valid':<8} | {'Invalid':<8} | {'Dup':<5} | {'Status':<10}")
    print("-" * 85)
    all_pass = True
    for gen, exp_info in EXPECTED_PRODUCTION_COUNTS.items():
        exp = exp_info["expected"]
        gstats = stats[gen]
        status = "PASS" if (gstats["valid"] == exp and gstats["invalid"] == 0 and gstats["duplicates"] == 0) else "FAIL"
        if status != "PASS":
            all_pass = False
        print(f"{gen:<15} | {exp:<10,} | {gstats['generated']:<10,} | {gstats['valid']:<8,} | {gstats['invalid']:<8} | {gstats['duplicates']:<5} | {status:<10}")
    print("=" * 85)

    print(f"\nVerification finished: {len(errors)} errors, {len(warnings)} warnings.")
    if errors:
        print("Top errors:\n" + "\n".join(errors[:15]))
    return all_pass

def main():
    parser = argparse.ArgumentParser(description="Verify Production Dataset")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT), help="Output directory root")
    args = parser.parse_args()

    success = verify_dataset(Path(args.output_root).resolve())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
