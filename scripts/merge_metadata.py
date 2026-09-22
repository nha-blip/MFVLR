#!/usr/bin/env python3
"""
MFVLR / GenFace-Reproduced: Metadata Shard Aggregator & Validator
Merges per-task metadata shards into a single unified CSV file per generator
and a combined dataset-wide master CSV file.
Validates completeness, sorts by sample_id, and checks for missing/duplicate samples.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "output")).resolve()

EXPECTED_COUNTS = {
    "StyleGAN3": 50000,
    "IAFaces": 5000,
    "LatTrans": 60000,
    "FaceSwapper": 30000,
    "DDPM": 50000,
    "LatDiff": 60000,
    "CollDiff": 50000,
    "DiffAE": 70000,
}

def merge_generator_shards(meta_dir: Path, gen: str, expected_count: int) -> List[Dict[str, str]]:
    gen_meta_dir = meta_dir / gen
    if not gen_meta_dir.exists():
        print(f"[{gen:<12}] No metadata directory found at {gen_meta_dir}")
        return []

    shard_files = sorted(gen_meta_dir.glob("shard_*.csv"))
    print(f"[{gen:<12}] Found {len(shard_files)} shard files.")

    all_records = []
    seen_ids = set()
    duplicates = []

    for sf in shard_files:
        with open(sf, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                sid = r.get("sample_id", "")
                if sid in seen_ids:
                    duplicates.append(sid)
                seen_ids.add(sid)
                all_records.append(r)

    # Sort by sample_id
    all_records.sort(key=lambda x: x.get("sample_id", ""))

    print(f"[{gen:<12}] Aggregated {len(all_records):,} records (Expected: {expected_count:,}).")
    if duplicates:
        print(f"[{gen:<12}] WARNING: Found {len(duplicates)} duplicate sample IDs!")
    if len(all_records) < expected_count:
        print(f"[{gen:<12}] WARNING: Missing {expected_count - len(all_records):,} samples!")
    elif len(all_records) == expected_count and not duplicates:
        print(f"[{gen:<12}] PASS: Complete and unique.")

    # Write generator-specific combined CSV
    out_csv = meta_dir / f"all_{gen.lower()}.csv"
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"[{gen:<12}] Saved combined CSV to {out_csv}")

    return all_records

def main():
    parser = argparse.ArgumentParser(description="Merge MFVLR Metadata Shards")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT), help="Output directory root")
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    meta_dir = out_root / "metadata"

    print("================================================================================")
    print("=== MFVLR / GenFace-Reproduced: Merging Metadata Shards ===")
    print("================================================================================")
    print(f"Scanning: {meta_dir}")

    master_records = []
    for gen, exp_count in EXPECTED_COUNTS.items():
        recs = merge_generator_shards(meta_dir, gen, exp_count)
        master_records.extend(recs)

    # Write master combined CSV
    master_csv = meta_dir / "all_production.csv"
    if master_records:
        fieldnames = list(master_records[0].keys())
        with open(master_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(master_records)
        print(f"\nSaved MASTER production CSV ({len(master_records):,} records) to: {master_csv}")

    print("================================================================================")
    print("Shard merge completed.")

if __name__ == "__main__":
    main()
