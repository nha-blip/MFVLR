#!/usr/bin/env python3
"""
run_pipeline.py - MFVLR Dataset Reproduction Master Pipeline Runner

Orchestrates the complete dataset reproduction workflow:
1. (Optional) Checkpoint & data download check (download_data.py).
2. Synthetic/mock or checkpoint-based generation:
   - generate_efs.py (DDPM, LatDiff, CollDiff, StyleGAN3)
   - generate_am.py (DiffAE, LatTrans, IAFaces)
   - generate_fs.py (DiffFace, FSLSD, FaceSwapper)
3. Scan dataset & create metadata (build_metadata.py).
4. Generate ground-truth masks (generate_masks.py).
5. Partition dataset with zero-leakage check (split_dataset.py).
6. Comprehensive protocol verification & visual reports (verify_dataset.py).

Flags:
--mock: Run end-to-end with synthetic mock samples (no heavy checkpoints needed).
--dry-run: Print all execution stages without generating files.
--count-per-generator: Number of samples to generate per generator (default: 5).
--protocol: Splitting protocol (default: cross_generator).
--seed: Master random seed.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def run_stage(cmd: List[str], desc: str, dry_run: bool = False) -> bool:
    """Executes a pipeline stage with logging and error handling."""
    logger.info("\n>>> [STAGE] %s", desc)
    logger.info("Command: %s", " ".join(cmd))

    if dry_run:
        logger.info("[DRY-RUN] Stage skipped.")
        return True

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error("Stage '%s' failed with return code %d", desc, e.returncode)
        return False
    except Exception as e:
        logger.error("Stage '%s' encountered unexpected error: %s", desc, e)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Master Pipeline Runner for MFVLR Dataset Reproduction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=str, default="MFVLR_Dataset", help="Root directory of MFVLR dataset")
    parser.add_argument("--count-per-generator", type=int, default=5, help="Number of samples to generate per generator")
    parser.add_argument("--protocol", type=str, default="cross_generator", help="Splitting protocol name")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock generation mode (default: True for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without creating files")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python binary path")

    args = parser.parse_args()
    py = args.python_bin
    root = Path(args.root)

    logger.info("================================================================")
    logger.info("MFVLR DATASET REPRODUCTION PIPELINE")
    logger.info("Dataset Root: %s", root)
    logger.info("Samples Per Generator: %d", args.count_per_generator)
    logger.info("Protocol: %s | Seed: %d | Mock: %s", args.protocol, args.seed, args.mock)
    logger.info("================================================================")

    mock_flag = ["--mock"] if args.mock else []
    dry_flag = ["--dry-run"] if args.dry_run else []

    # Stage 1: Generate EFS samples
    efs_generators = ["DDPM", "LatDiff", "CollDiff", "StyleGAN3"]
    for gen in efs_generators:
        cmd = [
            py, "generate_efs.py",
            "--generator", gen,
            "--count", str(args.count_per_generator),
            "--seed", str(args.seed),
            "--out-dir", str(root / "images" / "EFS"),
        ] + mock_flag + dry_flag
        if not run_stage(cmd, f"Generate EFS - {gen}", dry_run=args.dry_run):
            return 1

    # Stage 2: Generate AM samples
    am_generators = ["DiffAE", "LatTrans", "IAFaces"]
    for gen in am_generators:
        cmd = [
            py, "generate_am.py",
            "--generator", gen,
            "--source-dir", str(root / "images" / "real"),
            "--dataset-root", str(root),
            "--count", str(args.count_per_generator),
            "--seed", str(args.seed),
        ] + mock_flag + dry_flag
        if not run_stage(cmd, f"Generate AM - {gen}", dry_run=args.dry_run):
            return 1

    # Stage 3: Generate FS samples
    fs_generators = ["DiffFace", "FSLSD", "FaceSwapper"]
    for gen in fs_generators:
        cmd = [
            py, "generate_fs.py",
            "--generator", gen,
            "--source-dir", str(root / "images" / "real"),
            "--dataset-root", str(root),
            "--count", str(args.count_per_generator),
            "--seed", str(args.seed),
        ] + mock_flag + dry_flag
        if not run_stage(cmd, f"Generate FS - {gen}", dry_run=args.dry_run):
            return 1

    # Stage 4: Build metadata
    cmd = [
        py, "build_metadata.py",
        "--root", str(root),
        "--config", "configs/generators.yaml",
        "--output", str(root / "metadata" / "all.csv"),
    ] + dry_flag
    if not run_stage(cmd, "Build Metadata", dry_run=args.dry_run):
        return 1

    # Stage 5: Generate masks
    cmd = [
        py, "generate_masks.py",
        "--metadata", str(root / "metadata" / "all.csv"),
        "--root", str(root),
        "--threshold", "0.1",
        "--size", "224",
    ] + dry_flag
    if not run_stage(cmd, "Generate Masks", dry_run=args.dry_run):
        return 1

    # Stage 6: Split dataset with zero-leakage check
    cmd = [
        py, "split_dataset.py",
        "--metadata", str(root / "metadata" / "all.csv"),
        "--protocol-config", "configs/protocols.yaml",
        "--protocol-name", args.protocol,
        "--output-dir", str(root / "metadata"),
        "--seed", str(args.seed),
    ] + dry_flag
    if not run_stage(cmd, f"Split Dataset ({args.protocol})", dry_run=args.dry_run):
        return 1

    # Stage 7: Comprehensive verification
    cmd = [
        py, "verify_dataset.py",
        "--metadata", str(root / "metadata" / "all.csv"),
        "--root", str(root),
        "--report", str(root / "logs" / "dataset_report.json"),
        "--visualize", "3",
    ]
    if not run_stage(cmd, "Verify Dataset", dry_run=args.dry_run):
        return 1

    logger.info("\n================================================================")
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("Dataset ready for MFVLR training in: %s", root)
    logger.info("================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
