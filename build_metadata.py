#!/usr/bin/env python3
"""
build_metadata.py - MFVLR Dataset Metadata Builder

Scans the MFVLR_Dataset directory tree, validates source-fake pairs for AM/FS,
maps generators to taxonomy (EFS/AM/FS, GAN/Diffusion) via configs/generators.yaml,
generates hierarchical prompts (L1-L4) via PromptGenerator,
detects missing files/duplicates without silently dropping error samples,
and exports metadata/all.csv.

Protocol Compliance:
- Strict schema with provenance.
- REAL: label=0, forgery_type=REAL, architecture=REAL, generator=Real.
- AM/FS: requires corresponding source image.
- Does NOT randomly split dataset (split left unassigned unless protocol verified).
- Does NOT concatenate L1-L4 into a single prompt string.
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from prompt_generator import PromptGenerator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_metadata")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_generator_config(config_path: Path) -> Dict[str, Dict[str, Any]]:
    """Loads and validates generators.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Generators config not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid generators.yaml format. Expected dictionary.")

    if "generators" in config:
        return config["generators"]

    return config


class MetadataBuilder:
    """Builds and validates metadata for the MFVLR dataset."""

    def __init__(
        self,
        dataset_root: Path,
        config_path: Path,
        prompt_generator: Optional[PromptGenerator] = None,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.images_dir = self.dataset_root / "images"
        self.source_dir = self.dataset_root / "source"
        self.target_dir = self.dataset_root / "target"
        self.masks_dir = self.dataset_root / "masks"
        self.metadata_dir = self.dataset_root / "metadata"

        self.generator_cfg = load_generator_config(config_path)
        self.prompt_gen = prompt_generator or PromptGenerator(config_path=config_path)

        self.samples: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def scan_dataset(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Scans dataset directories, pairs fake with source images, and builds records."""
        self.samples.clear()
        self.errors.clear()
        self.warnings.clear()

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        seen_sample_ids: Set[str] = set()

        # 1. Scan REAL images: images/real/
        real_dir = self.images_dir / "real"
        if real_dir.exists():
            for img_path in sorted(real_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    sample_id = f"real_{img_path.stem}"
                    if sample_id in seen_sample_ids:
                        self.errors.append(f"Duplicate sample_id detected: {sample_id} ({img_path})")
                        continue
                    seen_sample_ids.add(sample_id)

                    rel_img = img_path.relative_to(self.dataset_root).as_posix()
                    rel_mask = (Path("masks") / "real" / f"{img_path.stem}.png").as_posix()

                    prompts = self.prompt_gen.generate(label=0, forgery_type="REAL", architecture="REAL", generator="Real")

                    self.samples.append({
                        "sample_id": sample_id,
                        "image_path": rel_img,
                        "source_image_path": "",
                        "target_image_path": "",
                        "mask_path": rel_mask,
                        "label": 0,
                        "forgery_type": "REAL",
                        "architecture": "REAL",
                        "generator": "Real",
                        "split": "",
                        "L1": prompts["L1"],
                        "L2": prompts["L2"],
                        "L3": prompts["L3"],
                        "L4": prompts["L4"],
                    })

        # 2. Scan Forgery groups: images/{EFS, AM, FS}/<generator>/
        for forgery_group in ["EFS", "AM", "FS"]:
            group_dir = self.images_dir / forgery_group
            if not group_dir.exists():
                logger.debug("Group directory does not exist yet: %s", group_dir)
                continue

            for gen_dir in sorted(group_dir.iterdir()):
                if not gen_dir.is_dir():
                    continue

                generator_name = gen_dir.name
                # Find matching config key (case-insensitive)
                matched_key = None
                for key in self.generator_cfg:
                    if key.lower() == generator_name.lower():
                        matched_key = key
                        break

                if not matched_key:
                    self.errors.append(
                        f"Generator '{generator_name}' under {forgery_group} is not defined in generators.yaml"
                    )
                    continue

                gen_info = self.generator_cfg[matched_key]
                cfg_forgery_type = gen_info.get("forgery_type")
                cfg_architecture = gen_info.get("architecture")
                needs_source = gen_info.get("needs_source", False)

                if cfg_forgery_type != forgery_group:
                    self.errors.append(
                        f"Folder hierarchy mismatch: generator '{matched_key}' is in '{forgery_group}' folder, "
                        f"but configured as '{cfg_forgery_type}' in generators.yaml"
                    )

                # Scan images in generator directory
                for img_path in sorted(gen_dir.iterdir()):
                    if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue

                    sample_id = f"{matched_key.lower()}_{img_path.stem}"
                    if sample_id in seen_sample_ids:
                        self.errors.append(f"Duplicate sample_id detected: {sample_id} ({img_path})")
                        continue
                    seen_sample_ids.add(sample_id)

                    rel_img = img_path.relative_to(self.dataset_root).as_posix()
                    rel_mask = (Path("masks") / forgery_group / matched_key / f"{img_path.stem}.png").as_posix()

                    rel_source = ""
                    rel_target = ""

                    if needs_source:
                        # Locate corresponding source image
                        # Expected: source/<forgery_group>/<generator>/<stem>.<ext>
                        gen_source_dir = self.source_dir / forgery_group / matched_key
                        source_candidate = None

                        if gen_source_dir.exists():
                            # Check same filename or matching stem
                            direct_match = gen_source_dir / img_path.name
                            if direct_match.exists():
                                source_candidate = direct_match
                            else:
                                for ext in IMAGE_EXTENSIONS:
                                    cand = gen_source_dir / f"{img_path.stem}{ext}"
                                    if cand.exists():
                                        source_candidate = cand
                                        break

                        if source_candidate is None:
                            self.errors.append(
                                f"Missing source image for {forgery_group}/{matched_key} sample: {img_path.name} "
                                f"(expected in {gen_source_dir})"
                            )
                        else:
                            rel_source = source_candidate.relative_to(self.dataset_root).as_posix()

                        # Optional target image for FS
                        if forgery_group == "FS":
                            gen_target_dir = self.target_dir / matched_key
                            if not gen_target_dir.exists():
                                gen_target_dir = self.target_dir / "FS" / matched_key

                            if gen_target_dir.exists():
                                direct_target = gen_target_dir / img_path.name
                                if direct_target.exists():
                                    rel_target = direct_target.relative_to(self.dataset_root).as_posix()
                                else:
                                    for ext in IMAGE_EXTENSIONS:
                                        cand = gen_target_dir / f"{img_path.stem}{ext}"
                                        if cand.exists():
                                            rel_target = cand.relative_to(self.dataset_root).as_posix()
                                            break

                    prompts = self.prompt_gen.generate(
                        label=1,
                        forgery_type=cfg_forgery_type,
                        architecture=cfg_architecture,
                        generator=matched_key,
                    )

                    self.samples.append({
                        "sample_id": sample_id,
                        "image_path": rel_img,
                        "source_image_path": rel_source,
                        "target_image_path": rel_target,
                        "mask_path": rel_mask,
                        "label": 1,
                        "forgery_type": cfg_forgery_type,
                        "architecture": cfg_architecture,
                        "generator": matched_key,
                        "split": "",
                        "L1": prompts["L1"],
                        "L2": prompts["L2"],
                        "L3": prompts["L3"],
                        "L4": prompts["L4"],
                    })

        return self.samples, self.errors

    def export_csv(self, output_path: Path, dry_run: bool = False) -> None:
        """Exports the scanned samples to CSV."""
        fieldnames = [
            "sample_id",
            "image_path",
            "source_image_path",
            "target_image_path",
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
        ]

        logger.info("Scanned %d total samples (%d errors detected)", len(self.samples), len(self.errors))

        if self.errors:
            logger.error("Dataset scanning encountered %d errors:", len(self.errors))
            for err in self.errors[:20]:
                logger.error("  - %s", err)
            if len(self.errors) > 20:
                logger.error("  ... and %d more errors.", len(self.errors) - 20)

        if dry_run:
            logger.info("[DRY-RUN] CSV export skipped. Would write to: %s", output_path)
            return

        if self.errors:
            logger.warning(
                "Proceeding with export despite errors. Note: downstream verification may fail until errors are resolved."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)

        logger.info("Successfully exported %d samples to %s", len(self.samples), output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MFVLR Dataset Metadata Builder. Scans dataset, validates pairs, and generates metadata/all.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        default="MFVLR_Dataset",
        help="Root directory of the dataset containing images/, source/, masks/, etc.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/generators.yaml",
        help="Path to generators.yaml configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="MFVLR_Dataset/metadata/all.csv",
        help="Destination path for all.csv metadata file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform scanning and validation without writing the CSV output.",
    )

    args = parser.parse_args()

    dataset_root = Path(args.root)
    config_path = Path(args.config)
    output_path = Path(args.output)

    try:
        builder = MetadataBuilder(dataset_root=dataset_root, config_path=config_path)
        builder.scan_dataset()
        builder.export_csv(output_path, dry_run=args.dry_run)
    except Exception as e:
        logger.error("Fatal error building metadata: %s", e, exc_info=True)
        return 1

    return 0 if not builder.errors else 2


if __name__ == "__main__":
    sys.exit(main())
