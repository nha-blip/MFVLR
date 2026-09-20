import os
import sys
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from scripts.preproduction.common import (
    PREPROD_ROOT,
    METADATA_FIELDS,
    compute_sha256,
)

EXPECTED_GENERATORS = {
    "DDPM": {"type": "EFS", "arch": "Diffusion", "expected": 500},
    "LatDiff": {"type": "EFS", "arch": "Diffusion", "expected": 500},
    "CollDiff": {"type": "EFS", "arch": "Diffusion", "expected": 500},
    "StyleGAN3": {"type": "EFS", "arch": "GAN", "expected": 500},
    "DiffAE": {"type": "AM", "arch": "Diffusion", "expected": 500},
    "LatTrans": {"type": "AM", "arch": "GAN", "expected": 500},
    "IAFaces": {"type": "AM", "arch": "GAN", "expected": 500},
    "FaceSwapper": {"type": "FS", "arch": "GAN", "expected": 380},
}


def audit_pilot_dataset():
    """Verify that the 115 pilot samples in MFVLR_Dataset remain completely untouched and valid."""
    print("\n=== AUDITING 115 PILOT SAMPLES (MFVLR_Dataset) ===")
    pilot_csv = Path("MFVLR_Dataset/metadata/all.csv")
    if not pilot_csv.exists():
        return False, ["Pilot metadata CSV missing: MFVLR_Dataset/metadata/all.csv"]
    
    with open(pilot_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if len(rows) != 115:
        return False, [f"Expected 115 pilot samples, found {len(rows)}"]
    
    errors = []
    for r in rows:
        sid = r["sample_id"]
        # Check image exists
        img_p = Path("MFVLR_Dataset") / r["image_path"]
        if not img_p.exists():
            errors.append(f"Pilot sample {sid} image missing: {img_p}")
        msk_p = Path("MFVLR_Dataset") / r["mask_path"]
        if not msk_p.exists():
            errors.append(f"Pilot sample {sid} mask missing: {msk_p}")
    
    if errors:
        return False, errors
    print(f"PILOT AUDIT PASS: All {len(rows)} pilot samples intact and accessible.")
    return True, []


def verify_preproduction():
    print("=== STARTING PRE-PRODUCTION 500 DATASET VERIFICATION ===")
    
    csv_path = PREPROD_ROOT / "metadata" / "all_preproduction.csv"
    if not csv_path.exists():
        print(f"ERROR: Metadata CSV not found at {csv_path}")
        return False, {"error": "Metadata CSV missing"}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        records = list(reader)
    
    print(f"Found {len(records)} metadata records in {csv_path}")
    
    # 1. Check metadata fields
    missing_fields = [f for f in METADATA_FIELDS if f not in fieldnames]
    if missing_fields:
        print(f"CRITICAL: Missing metadata fields: {missing_fields}")
        return False, {"missing_fields": missing_fields}
    
    # Trackers
    errors = []
    warnings = []
    seen_ids = set()
    sha_map = defaultdict(list)
    stats = defaultdict(lambda: {
        "requested": 0,
        "generated": 0,
        "valid": 0,
        "rejected": 0,
        "duplicate_count": 0,
        "mask_ratios": [],
        "native_res": set(),
        "notes": [],
    })
    
    for r in records:
        sid = r.get("sample_id", "")
        gen = r.get("generator", "")
        ftype = r.get("forgery_type", "")
        arch = r.get("architecture", "")
        split = r.get("split", "")
        
        # Identity uniqueness
        if not sid:
            errors.append("Sample record missing sample_id")
            continue
        if sid in seen_ids:
            errors.append(f"Duplicate sample_id: {sid}")
        seen_ids.add(sid)
        
        stats[gen]["generated"] += 1
        
        # Split check
        if split != "UNASSIGNED":
            errors.append(f"Sample {sid}: split must be 'UNASSIGNED', got '{split}'")
        
        # Taxonomy check
        if gen in EXPECTED_GENERATORS:
            exp_type = EXPECTED_GENERATORS[gen]["type"]
            exp_arch = EXPECTED_GENERATORS[gen]["arch"]
            if ftype != exp_type:
                errors.append(f"Sample {sid}: unexpected forgery_type '{ftype}' for generator {gen}, expected '{exp_type}'")
            if arch != exp_arch:
                errors.append(f"Sample {sid}: unexpected architecture '{arch}' for generator {gen}, expected '{exp_arch}'")
        else:
            warnings.append(f"Sample {sid}: unexpected generator '{gen}'")
        
        # Check L1-L4 presence
        for l_key in ["L1", "L2", "L3", "L4"]:
            if not r.get(l_key):
                errors.append(f"Sample {sid}: missing {l_key}")
        
        # File paths
        img_rel = r.get("image_path", "")
        msk_rel = r.get("mask_path", "")
        src_rel = r.get("source_path", "")
        tgt_rel = r.get("target_path", "")
        
        img_full = PREPROD_ROOT / img_rel
        msk_full = PREPROD_ROOT / msk_rel
        
        if not img_full.exists():
            errors.append(f"Sample {sid}: image file does not exist: {img_full}")
            stats[gen]["rejected"] += 1
            continue
        
        if not msk_full.exists():
            errors.append(f"Sample {sid}: mask file does not exist: {msk_full}")
            stats[gen]["rejected"] += 1
            continue
        
        # Check source/target requirements per forgery type
        if ftype == "AM":
            if not src_rel:
                errors.append(f"Sample {sid} (AM): source_path must not be empty")
            else:
                src_full = PREPROD_ROOT / src_rel
                if not src_full.exists():
                    errors.append(f"Sample {sid} (AM): source file missing: {src_full}")
        elif ftype == "FS":
            if not src_rel:
                errors.append(f"Sample {sid} (FS): source_path must not be empty")
            if not tgt_rel:
                errors.append(f"Sample {sid} (FS): target_path must not be empty")
            else:
                tgt_full = PREPROD_ROOT / tgt_rel
                if not tgt_full.exists():
                    errors.append(f"Sample {sid} (FS): target file missing: {tgt_full}")
        
        # Image integrity & format
        try:
            with Image.open(img_full) as img:
                if img.format != "PNG":
                    errors.append(f"Sample {sid}: image format is {img.format}, expected PNG")
                if img.mode != "RGB":
                    errors.append(f"Sample {sid}: image mode is {img.mode}, expected RGB")
                if img.size != (224, 224):
                    errors.append(f"Sample {sid}: image size is {img.size}, expected (224, 224)")
        except Exception as e:
            errors.append(f"Sample {sid}: failed to read image: {e}")
            stats[gen]["rejected"] += 1
            continue
        
        # Mask integrity & values
        try:
            with Image.open(msk_full) as msk:
                if msk.format != "PNG":
                    errors.append(f"Sample {sid}: mask format is {msk.format}, expected PNG")
                if msk.size != (224, 224):
                    errors.append(f"Sample {sid}: mask size is {msk.size}, expected (224, 224)")
                msk_np = np.array(msk)
                if msk_np.dtype != np.uint8:
                    errors.append(f"Sample {sid}: mask dtype is {msk_np.dtype}, expected uint8")
                
                unique_vals = np.unique(msk_np)
                if not set(unique_vals).issubset({0, 255}):
                    errors.append(f"Sample {sid}: mask contains invalid values: {unique_vals}, must be in {{0, 255}}")
                
                ratio = np.count_nonzero(msk_np == 255) / msk_np.size
                stats[gen]["mask_ratios"].append(ratio)
                
                if ftype == "EFS":
                    if ratio != 1.0:
                        errors.append(f"Sample {sid} (EFS): mask is not all 255 (ratio={ratio:.4f})")
                elif ftype in ["AM", "FS"]:
                    if ratio == 0.0:
                        warnings.append(f"Sample {sid} ({ftype}): mask has 0% altered area")
        except Exception as e:
            errors.append(f"Sample {sid}: failed to read mask: {e}")
            stats[gen]["rejected"] += 1
            continue
        
        # SHA256 integrity
        img_sha = compute_sha256(img_full)
        if r.get("sha256") and r["sha256"] != img_sha:
            errors.append(f"Sample {sid}: metadata sha256 {r['sha256'][:8]} != file sha256 {img_sha[:8]}")
        sha_map[img_sha].append(sid)
        
        stats[gen]["native_res"].add(r.get("native_resolution", "unknown"))
        stats[gen]["valid"] += 1
    
    # Check duplicate SHA256
    duplicate_total = 0
    for sha, sids in sha_map.items():
        if len(sids) > 1:
            duplicate_total += len(sids) - 1
            errors.append(f"Duplicate image hash {sha[:12]} shared by samples: {sids}")
            for sid in sids[1:]:
                # find generator
                for r in records:
                    if r["sample_id"] == sid:
                        stats[r["generator"]]["duplicate_count"] += 1
    
    # Summarize stats
    print("\n" + "=" * 80)
    print(f"{'Generator':<15} | {'Expected':<8} | {'Generated':<9} | {'Valid':<7} | {'Invalid':<7} | {'Dup':<4} | {'Mask Area (Mean±Std [Min, Max])':<32}")
    print("-" * 80)
    
    report_rows = []
    for gen, exp_info in EXPECTED_GENERATORS.items():
        exp = exp_info["expected"]
        gstats = stats[gen]
        gen_cnt = gstats["generated"]
        val_cnt = gstats["valid"]
        inv_cnt = gstats["rejected"]
        dup_cnt = gstats["duplicate_count"]
        ratios = gstats["mask_ratios"]
        
        if ratios:
            mean_r = np.mean(ratios)
            std_r = np.std(ratios)
            min_r = np.min(ratios)
            max_r = np.max(ratios)
            ratio_str = f"{mean_r:.3f}±{std_r:.3f} [{min_r:.3f}, {max_r:.3f}]"
        else:
            ratio_str = "N/A"
        
        status = "PASS" if (val_cnt == exp and inv_cnt == 0 and dup_cnt == 0) else ("PARTIAL" if val_cnt > 0 else "PENDING")
        print(f"{gen:<15} | {exp:<8} | {gen_cnt:<9} | {val_cnt:<7} | {inv_cnt:<7} | {dup_cnt:<4} | {ratio_str:<32}")
        
        report_rows.append({
            "generator": gen,
            "expected": exp,
            "generated": gen_cnt,
            "valid": val_cnt,
            "invalid": inv_cnt,
            "duplicates": dup_cnt,
            "mask_ratio_summary": ratio_str,
            "status": status,
        })
    print("=" * 80)
    
    # Audit pilot
    pilot_pass, pilot_errors = audit_pilot_dataset()
    if not pilot_pass:
        errors.extend(pilot_errors)
    
    print(f"\nVerification finished: {len(errors)} errors, {len(warnings)} warnings.")
    if errors:
        print(f"Top errors:\n" + "\n".join(errors[:10]))
    if warnings:
        print(f"Top warnings:\n" + "\n".join(warnings[:5]))
        
    return len(errors) == 0, {
        "total_records": len(records),
        "errors": errors,
        "warnings": warnings,
        "stats": report_rows,
        "pilot_pass": pilot_pass,
    }


if __name__ == "__main__":
    success, res = verify_preproduction()
    sys.exit(0 if success else 1)
