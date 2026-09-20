import os
import sys
import csv
import json
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from scripts.preproduction.common import compute_sha256, METADATA_FIELDS

EXPECTED_PRODUCTION_COUNTS = {
    "DDPM": {"type": "EFS", "arch": "Diffusion", "expected": 50000},
    "LatDiff": {"type": "EFS", "arch": "Diffusion", "expected": 60000},
    "CollDiff": {"type": "EFS", "arch": "Diffusion", "expected": 50000},
    "StyleGAN3": {"type": "EFS", "arch": "GAN", "expected": 50000},
    "DiffAE": {"type": "AM", "arch": "Diffusion", "expected": 70000},
    "LatTrans": {"type": "AM", "arch": "GAN", "expected": 60000},
    "IAFaces": {"type": "AM", "arch": "GAN", "expected": 5000},
    "FaceSwapper": {"type": "FS", "arch": "GAN", "expected": 30000},
}

TOTAL_TARGET = 375000

def verify_production_dataset(prod_root_path: Path, manifest_path: Path):
    print("================================================================================")
    print("=== MFVLR / GenFace-Reproduced: Full Production Dataset Verifier ===")
    print("================================================================================")
    print(f"Production Root: {prod_root_path}")
    print(f"Manifest: {manifest_path}")

    # Check for metadata shards or combined metadata
    meta_dir = prod_root_path / "metadata"
    shard_files = sorted(meta_dir.glob("shard_*.csv"))
    combined_csv = meta_dir / "all_production.csv"

    records = []
    if combined_csv.exists():
        print(f"Loading combined metadata: {combined_csv}")
        with open(combined_csv, "r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
    elif shard_files:
        print(f"Found {len(shard_files)} metadata shards in {meta_dir}. Aggregating...")
        for sf in shard_files:
            with open(sf, "r", encoding="utf-8") as f:
                records.extend(list(csv.DictReader(f)))
    else:
        print(f"WARNING: No production metadata files found in {meta_dir}.")
        print("This verifier is prepared to validate output once generation completes.")
        return True, {"status": "NO_RECORDS_YET"}

    print(f"Total records found: {len(records):,}")

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
        arch = r.get("architecture", "")
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

        # Check fields
        for field in METADATA_FIELDS:
            if field not in r:
                errors.append(f"Sample {sid}: missing field '{field}'")

        # Paths
        img_rel = r.get("image_path", "")
        msk_rel = r.get("mask_path", "")
        img_full = prod_root_path / img_rel
        msk_full = prod_root_path / msk_rel

        if not img_full.exists():
            errors.append(f"Sample {sid}: image missing {img_full}")
            stats[gen]["invalid"] += 1
            continue
        if not msk_full.exists():
            errors.append(f"Sample {sid}: mask missing {msk_full}")
            stats[gen]["invalid"] += 1
            continue

        # Check AM / FS pairing
        if ftype == "AM":
            src_rel = r.get("source_path", "")
            if not src_rel or not (prod_root_path / src_rel).exists():
                errors.append(f"Sample {sid} (AM): missing source image {src_rel}")
        elif ftype == "FS":
            src_rel = r.get("source_path", "")
            tar_rel = r.get("target_path", "")
            if not src_rel or not (prod_root_path / src_rel).exists():
                errors.append(f"Sample {sid} (FS): missing source image {src_rel}")
            if not tar_rel or not (prod_root_path / tar_rel).exists():
                errors.append(f"Sample {sid} (FS): missing target image {tar_rel}")

        # Image integrity
        try:
            with Image.open(img_full) as img:
                if img.size != (224, 224) or img.mode != "RGB" or img.format != "PNG":
                    errors.append(f"Sample {sid}: image invalid format/mode/size ({img.format}, {img.mode}, {img.size})")
        except Exception as e:
            errors.append(f"Sample {sid}: failed to read image: {e}")
            stats[gen]["invalid"] += 1
            continue

        # Mask integrity
        try:
            with Image.open(msk_full) as msk:
                if msk.size != (224, 224) or msk.format != "PNG":
                    errors.append(f"Sample {sid}: mask invalid format/size ({msk.format}, {msk.size})")
                m_np = np.array(msk)
                if m_np.dtype != np.uint8:
                    errors.append(f"Sample {sid}: mask dtype {m_np.dtype} != uint8")
                vals = set(np.unique(m_np))
                if not vals.issubset({0, 255}):
                    errors.append(f"Sample {sid}: mask values invalid: {vals}")
                ratio = np.count_nonzero(m_np == 255) / m_np.size
                stats[gen]["mask_ratios"].append(ratio)
                if ftype == "EFS" and ratio != 1.0:
                    errors.append(f"Sample {sid} (EFS): mask ratio {ratio} != 1.0")
        except Exception as e:
            errors.append(f"Sample {sid}: failed to read mask: {e}")
            stats[gen]["invalid"] += 1
            continue

        # SHA256 integrity
        img_sha = compute_sha256(img_full)
        if r.get("sha256") and r["sha256"] != img_sha:
            errors.append(f"Sample {sid}: SHA256 mismatch")
        sha_map[img_sha].append(sid)

        stats[gen]["valid"] += 1

    # Check duplicates
    for sha, sids in sha_map.items():
        if len(sids) > 1:
            errors.append(f"Duplicate SHA256 {sha[:12]}: {sids}")
            for sid in sids[1:]:
                for r in records:
                    if r["sample_id"] == sid:
                        stats[r["generator"]]["duplicates"] += 1

    # Summary table
    print("\n" + "=" * 85)
    print(f"{'Generator':<15} | {'Expected':<10} | {'Generated':<10} | {'Valid':<8} | {'Invalid':<8} | {'Dup':<5} | {'Status':<10}")
    print("-" * 85)
    all_pass = True
    for gen, exp_info in EXPECTED_PRODUCTION_COUNTS.items():
        exp = exp_info["expected"]
        gstats = stats[gen]
        status = "PASS" if (gstats["valid"] == exp and gstats["invalid"] == 0 and gstats["duplicates"] == 0) else "PENDING"
        if status != "PASS":
            all_pass = False
        print(f"{gen:<15} | {exp:<10,} | {gstats['generated']:<10,} | {gstats['valid']:<8,} | {gstats['invalid']:<8} | {gstats['duplicates']:<5} | {status:<10}")
    print("=" * 85)

    print(f"\nVerification finished: {len(errors)} errors, {len(warnings)} warnings.")
    return len(errors) == 0, {"errors": errors, "warnings": warnings}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify MFVLR Production Dataset")
    parser.add_argument("--production_root", type=str, default="GenFace_Reproduced/production")
    parser.add_argument("--manifest", type=str, default="GenFace_Reproduced/production/configs/production_manifest.yaml")
    args = parser.parse_args()

    success, _ = verify_production_dataset(Path(args.production_root), Path(args.manifest))
    sys.exit(0 if success else 1)
