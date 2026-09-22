#!/usr/bin/env python3
"""
split_dataset.py - MFVLR Protocol-driven Dataset Splitter

Features:
- Config-driven protocol splitting (cross_generator, cross_architecture, cross_forgery, stratified_ratio).
- Prevents data leakage: verifies that source images/identities do not overlap across train, val, and test splits.
- Preserves all samples without dropping any.
- Deterministic with random seed.
- Exports metadata/train.csv, metadata/val.csv, metadata/test.csv and updates metadata/all.csv.
- CLI with --help and --dry-run.
"""

import argparse
import csv
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("split_dataset")


def load_protocol_config(config_path: Path, protocol_name: str) -> Dict[str, Any]:
    """Loads and validates protocol configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Protocol config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "protocols" not in data:
        raise ValueError("Invalid protocols.yaml format. Expected 'protocols' mapping at root.")

    protocols = data["protocols"]
    if protocol_name not in protocols:
        raise ValueError(
            f"Protocol '{protocol_name}' not found in {config_path}. "
            f"Available protocols: {list(protocols.keys())}"
        )

    return protocols[protocol_name]


class DatasetSplitter:
    """Partitions dataset samples according to specified protocol and verifies zero data leakage."""

    def __init__(
        self,
        metadata_path: Path,
        protocol: Dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.metadata_path = metadata_path.resolve()
        self.protocol = protocol
        self.seed = seed
        self.rng = random.Random(seed)

    def load_metadata(self) -> List[Dict[str, Any]]:
        """Loads metadata rows from CSV."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def check_leakage(self, splits: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Checks for source image / identity overlap between splits."""
        leakage_errors: List[str] = []
        source_sets: Dict[str, Set[str]] = defaultdict(set)

        for split_name, rows in splits.items():
            for r in rows:
                src = r.get("source_image_path", "").strip() or r.get("source_path", "").strip()
                if src:
                    source_sets[split_name].add(src)

        split_names = list(splits.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                overlap = source_sets[s1] & source_sets[s2]
                if overlap:
                    leakage_errors.append(
                        f"Data leakage detected between '{s1}' and '{s2}': {len(overlap)} shared source images! "
                        f"Sample overlaps: {list(overlap)[:5]}"
                    )

        return leakage_errors

    def split(self) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Executes splitting according to protocol."""
        rows = self.load_metadata()
        p_type = self.protocol.get("type", "generator_split")

        splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        all_updated: List[Dict[str, Any]] = []

        if p_type == "generator_split":
            train_gens = set(self.protocol.get("train_generators", []))
            val_gens = set(self.protocol.get("val_generators", []))
            test_gens = set(self.protocol.get("test_generators", []))
            real_ratios = self.protocol.get("real_split_ratio", {"train": 0.7, "val": 0.1, "test": 0.2})

            real_rows = []
            for r in rows:
                row_copy = dict(r)
                gen = row_copy.get("generator", "").strip()
                label = int(row_copy.get("label", 0))

                if label == 0 or gen == "Real":
                    real_rows.append(row_copy)
                elif gen in train_gens:
                    row_copy["split"] = "train"
                    splits["train"].append(row_copy)
                    all_updated.append(row_copy)
                elif gen in val_gens:
                    row_copy["split"] = "val"
                    splits["val"].append(row_copy)
                    all_updated.append(row_copy)
                elif gen in test_gens:
                    row_copy["split"] = "test"
                    splits["test"].append(row_copy)
                    all_updated.append(row_copy)
                else:
                    raise ValueError(
                        f"Generator '{gen}' is not assigned to any split in protocol config!"
                    )

            # Split Real samples
            self._split_samples_by_ratio(real_rows, real_ratios, splits, all_updated)

        elif p_type == "architecture_split":
            train_archs = set(self.protocol.get("train_architectures", []))
            val_gens = set(self.protocol.get("val_generators", []))
            test_archs = set(self.protocol.get("test_architectures", []))
            real_ratios = self.protocol.get("real_split_ratio", {"train": 0.7, "val": 0.1, "test": 0.2})

            real_rows = []
            for r in rows:
                row_copy = dict(r)
                arch = row_copy.get("architecture", "").strip()
                gen = row_copy.get("generator", "").strip()
                label = int(row_copy.get("label", 0))

                if label == 0 or arch == "REAL":
                    real_rows.append(row_copy)
                elif gen in val_gens:
                    row_copy["split"] = "val"
                    splits["val"].append(row_copy)
                    all_updated.append(row_copy)
                elif arch in train_archs:
                    row_copy["split"] = "train"
                    splits["train"].append(row_copy)
                    all_updated.append(row_copy)
                elif arch in test_archs:
                    row_copy["split"] = "test"
                    splits["test"].append(row_copy)
                    all_updated.append(row_copy)
                else:
                    raise ValueError(f"Architecture '{arch}' not covered by protocol.")

            self._split_samples_by_ratio(real_rows, real_ratios, splits, all_updated)

        elif p_type == "forgery_split":
            train_types = set(self.protocol.get("train_forgery_types", []))
            val_gens = set(self.protocol.get("val_generators", []))
            test_types = set(self.protocol.get("test_forgery_types", []))
            real_ratios = self.protocol.get("real_split_ratio", {"train": 0.7, "val": 0.1, "test": 0.2})

            real_rows = []
            for r in rows:
                row_copy = dict(r)
                ftype = row_copy.get("forgery_type", "").strip()
                gen = row_copy.get("generator", "").strip()
                label = int(row_copy.get("label", 0))

                if label == 0 or ftype == "REAL":
                    real_rows.append(row_copy)
                elif gen in val_gens:
                    row_copy["split"] = "val"
                    splits["val"].append(row_copy)
                    all_updated.append(row_copy)
                elif ftype in train_types:
                    row_copy["split"] = "train"
                    splits["train"].append(row_copy)
                    all_updated.append(row_copy)
                elif ftype in test_types:
                    row_copy["split"] = "test"
                    splits["test"].append(row_copy)
                    all_updated.append(row_copy)
                else:
                    raise ValueError(f"Forgery type '{ftype}' not covered by protocol.")

            self._split_samples_by_ratio(real_rows, real_ratios, splits, all_updated)

        elif p_type == "ratio_split":
            ratios = self.protocol.get("ratios", {"train": 0.7, "val": 0.1, "test": 0.2})
            enforce_grouping = self.protocol.get("enforce_source_grouping", True)

            # Group by generator
            by_generator: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in rows:
                by_generator[r.get("generator", "unknown")].append(dict(r))

            for gen_name, gen_rows in by_generator.items():
                if enforce_grouping:
                    # Group by source_image_path if present, else sample_id
                    source_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                    for r in gen_rows:
                        src_key = r.get("source_image_path", "").strip() or r.get("sample_id")
                        source_groups[src_key].append(r)

                    group_keys = list(source_groups.keys())
                    self.rng.shuffle(group_keys)

                    n = len(group_keys)
                    n_train = int(n * ratios.get("train", 0.7))
                    n_val = int(n * ratios.get("val", 0.1))

                    train_keys = set(group_keys[:n_train])
                    val_keys = set(group_keys[n_train : n_train + n_val])
                    test_keys = set(group_keys[n_train + n_val :])

                    for k, grp_rows in source_groups.items():
                        if k in train_keys:
                            target_split = "train"
                        elif k in val_keys:
                            target_split = "val"
                        else:
                            target_split = "test"

                        for r in grp_rows:
                            r["split"] = target_split
                            splits[target_split].append(r)
                            all_updated.append(r)
                else:
                    self._split_samples_by_ratio(gen_rows, ratios, splits, all_updated)

        else:
            raise ValueError(f"Unsupported protocol type: '{p_type}'")

        # Verify no data leakage
        leakage_errors = self.check_leakage(splits)
        if leakage_errors:
            logger.error("Splitting failed due to data leakage:")
            for err in leakage_errors:
                logger.error("  - %s", err)
            raise RuntimeError(f"Data leakage detected in {len(leakage_errors)} split pairs.")

        return splits, all_updated

    def _split_samples_by_ratio(
        self,
        samples: List[Dict[str, Any]],
        ratios: Dict[str, float],
        splits: Dict[str, List[Dict[str, Any]]],
        all_updated: List[Dict[str, Any]],
    ) -> None:
        """Splits a list of samples deterministically by ratio."""
        if not samples:
            return

        shuffled = list(samples)
        self.rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * ratios.get("train", 0.70))
        n_val = int(n * ratios.get("val", 0.10))

        train_part = shuffled[:n_train]
        val_part = shuffled[n_train : n_train + n_val]
        test_part = shuffled[n_train + n_val :]

        for r in train_part:
            r["split"] = "train"
            splits["train"].append(r)
            all_updated.append(r)

        for r in val_part:
            r["split"] = "val"
            splits["val"].append(r)
            all_updated.append(r)

        for r in test_part:
            r["split"] = "test"
            splits["test"].append(r)
            all_updated.append(r)


def export_split_csvs(
    splits: Dict[str, List[Dict[str, Any]]],
    all_rows: List[Dict[str, Any]],
    metadata_dir: Path,
    dry_run: bool = False,
) -> None:
    """Exports train.csv, val.csv, test.csv and updates all.csv."""
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

    logger.info(
        "Split summary: train=%d, val=%d, test=%d (total=%d)",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        len(all_rows),
    )

    if dry_run:
        logger.info("[DRY-RUN] CSV export skipped.")
        return

    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Update all.csv
    all_path = metadata_dir / "all.csv"
    with open(all_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("Updated %s with split labels", all_path)

    # 2. Write split CSVs
    for s_name, s_rows in splits.items():
        s_path = metadata_dir / f"{s_name}.csv"
        with open(s_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(s_rows)
        logger.info("Exported %d samples to %s", len(s_rows), s_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MFVLR Protocol-driven Dataset Splitter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="MFVLR_Dataset/metadata/all.csv",
        help="Path to metadata all.csv file.",
    )
    parser.add_argument(
        "--protocol-config",
        type=str,
        default="configs/protocols.yaml",
        help="Path to protocols.yaml configuration.",
    )
    parser.add_argument(
        "--protocol-name",
        type=str,
        default="cross_generator",
        help="Name of the protocol defined in protocols.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="MFVLR_Dataset/metadata",
        help="Output directory for split CSV files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform split calculation and leakage checks without writing CSVs.",
    )

    args = parser.parse_args()

    meta_path = Path(args.metadata)
    proto_path = Path(args.protocol_config)
    output_dir = Path(args.output_dir)

    try:
        protocol = load_protocol_config(proto_path, args.protocol_name)
        splitter = DatasetSplitter(metadata_path=meta_path, protocol=protocol, seed=args.seed)
        splits, all_rows = splitter.split()
        export_split_csvs(splits, all_rows, metadata_dir=output_dir, dry_run=args.dry_run)
        return 0
    except Exception as e:
        logger.error("Splitting failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
