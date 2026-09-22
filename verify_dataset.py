#!/usr/bin/env python3
"""
verify_dataset.py - MFVLR Dataset Comprehensive Verification Tool

Validates dataset against the MFVLR reproduction protocol:
1. File existence & readability (image, mask, source for AM/FS).
2. Image & mask resolution: strictly 224x224.
3. Mask values: strictly binary uint8 {0, 255}.
4. REAL integrity: mask is all zeros.
5. EFS integrity: mask is all 255.
6. AM/FS integrity: source exists, mask is not all 0 and not all 255 (localized forgery).
7. Hierarchical Prompts: L1-L4 present, non-empty, and NOT concatenated.
8. Metadata uniqueness: no duplicate sample_id.
9. Exports detailed dataset_report.json.
10. Optional: generates visual verification image grids.
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("verify_dataset")


def load_image_rgb(path: Path) -> Optional[np.ndarray]:
    """Load image safely on Windows with Unicode path support."""
    try:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))
    except Exception:
        return None


def load_mask_l(path: Path) -> Optional[np.ndarray]:
    """Load grayscale mask safely on Windows with Unicode path support."""
    try:
        with Image.open(path) as img:
            return np.array(img.convert("L"))
    except Exception:
        return None


def save_image_rgb(path: Path, arr: np.ndarray) -> None:
    """Save RGB/BGR image array safely on Windows with Unicode path support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")


class DatasetVerifier:
    """Performs rigorous verification of the MFVLR dataset."""

    def __init__(
        self,
        metadata_csv: Path,
        dataset_root: Optional[Path] = None,
        report_path: Optional[Path] = None,
        visualize_count: int = 5,
        visualize_dir: Optional[Path] = None,
    ) -> None:
        self.metadata_csv = metadata_csv.resolve()
        self.dataset_root = (dataset_root or metadata_csv.parent.parent).resolve()
        self.report_path = (report_path or (self.dataset_root / "logs" / "dataset_report.json")).resolve()
        self.visualize_count = visualize_count
        self.visualize_dir = (visualize_dir or (self.dataset_root / "logs" / "verification_samples")).resolve()

        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {
            "total_samples": 0,
            "real_samples": 0,
            "efs_samples": 0,
            "am_samples": 0,
            "fs_samples": 0,
            "by_generator": Counter(),
            "by_architecture": Counter(),
            "by_forgery_type": Counter(),
            "by_split": Counter(),
        }

    def _resolve_path(self, rel_or_abs: str) -> Optional[Path]:
        if not rel_or_abs or not str(rel_or_abs).strip():
            return None
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return self.dataset_root / p

    def verify(self) -> Dict[str, Any]:
        """Executes full verification protocol on metadata and associated files."""
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_csv}")

        seen_ids: Set[str] = set()
        samples_for_viz: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        with open(self.metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.stats["total_samples"] = len(rows)
        logger.info("Verifying %d samples from %s...", len(rows), self.metadata_csv)

        for idx, row in enumerate(rows):
            sample_id = row.get("sample_id", f"row_{idx}")
            img_path_str = row.get("image_path", "")
            mask_path_str = row.get("mask_path", "")
            source_path_str = row.get("source_image_path", "") or row.get("source_path", "")
            target_path_str = row.get("target_image_path", "") or row.get("target_path", "")
            label_str = row.get("label", "")
            forgery_type = row.get("forgery_type", "").strip()
            architecture = row.get("architecture", "").strip()
            generator = row.get("generator", "").strip()
            split = row.get("split", "").strip() or "unassigned"

            # 1. Duplicate sample_id check
            if sample_id in seen_ids:
                self.errors.append({
                    "sample_id": sample_id,
                    "type": "DUPLICATE_SAMPLE_ID",
                    "detail": f"Sample ID '{sample_id}' appears more than once in metadata",
                })
            seen_ids.add(sample_id)

            # Update stats
            self.stats["by_generator"][generator] += 1
            self.stats["by_architecture"][architecture] += 1
            self.stats["by_forgery_type"][forgery_type] += 1
            self.stats["by_split"][split] += 1

            # 2. Label validation
            try:
                label = int(label_str)
            except ValueError:
                self.errors.append({
                    "sample_id": sample_id,
                    "type": "INVALID_LABEL",
                    "detail": f"Label '{label_str}' cannot be parsed as integer",
                })
                label = -1

            if label == 0:
                self.stats["real_samples"] += 1
                if forgery_type != "REAL":
                    self.errors.append({
                        "sample_id": sample_id,
                        "type": "FORGERY_TYPE_MISMATCH",
                        "detail": f"Label is 0 (REAL) but forgery_type is '{forgery_type}'",
                    })
            elif label == 1:
                if forgery_type == "EFS":
                    self.stats["efs_samples"] += 1
                elif forgery_type == "AM":
                    self.stats["am_samples"] += 1
                elif forgery_type == "FS":
                    self.stats["fs_samples"] += 1
                else:
                    self.errors.append({
                        "sample_id": sample_id,
                        "type": "INVALID_FORGERY_TYPE",
                        "detail": f"Fake image has invalid forgery_type: '{forgery_type}'",
                    })
            else:
                self.errors.append({
                    "sample_id": sample_id,
                    "type": "INVALID_LABEL_VALUE",
                    "detail": f"Label must be 0 or 1, got {label}",
                })

            # 3. Prompt verification (L1-L4 must not be concatenated or empty)
            l1 = row.get("L1", "") or row.get("prompt_L1", "")
            l2 = row.get("L2", "") or row.get("prompt_L2", "")
            l3 = row.get("L3", "") or row.get("prompt_L3", "")
            l4 = row.get("L4", "") or row.get("prompt_L4", "")

            for lvl, text in [("L1", l1), ("L2", l2), ("L3", l3), ("L4", l4)]:
                if not text or not text.strip():
                    self.errors.append({
                        "sample_id": sample_id,
                        "type": "MISSING_PROMPT",
                        "detail": f"Prompt level {lvl} is empty",
                    })

            # Check concatenation rule: L1 must not contain L2/L3/L4 text
            if len(l1.split()) > 12:  # Expected "A photo of a fake face" or "A photo of a real face"
                self.warnings.append({
                    "sample_id": sample_id,
                    "type": "SUSPECT_CONCATENATED_PROMPT",
                    "detail": f"Prompt L1 has {len(l1.split())} words, may be concatenated: '{l1[:50]}...'",
                })

            # 4. Image file check
            img_file = self._resolve_path(img_path_str)
            if not img_file or not img_file.exists():
                self.errors.append({
                    "sample_id": sample_id,
                    "type": "MISSING_IMAGE_FILE",
                    "detail": f"Image file not found: {img_path_str}",
                })
                continue

            img = load_image_rgb(img_file)
            if img is None:
                self.errors.append({
                    "sample_id": sample_id,
                    "type": "UNREADABLE_IMAGE",
                    "detail": f"Failed to read image with PIL: {img_file}",
                })
                continue

            h, w, c = img.shape
            if (h, w) != (224, 224):
                self.errors.append({
                    "sample_id": sample_id,
                    "type": "WRONG_IMAGE_DIMENSIONS",
                    "detail": f"Image dimensions are {w}x{h}, strictly required 224x224",
                })

            # 5. Mask file check
            mask_file = self._resolve_path(mask_path_str)
            if not mask_file or not mask_file.exists():
                self.errors.append({
                    "sample_id": sample_id,
                    "type": "MISSING_MASK_FILE",
                    "detail": f"Mask file not found: {mask_path_str}",
                })
                continue

            mask = load_mask_l(mask_file)
            if mask is None:
                self.errors.append({
                    "sample_id": sample_id,
                    "type": "UNREADABLE_MASK",
                    "detail": f"Failed to read mask with PIL: {mask_file}",
                })
                continue

            if mask.ndim == 3:
                mask = mask[:, :, 0]

            mh, mw = mask.shape
            if (mh, mw) != (224, 224):
                self.errors.append({
                    "sample_id": sample_id,
                    "type": "WRONG_MASK_DIMENSIONS",
                    "detail": f"Mask dimensions are {mw}x{mh}, strictly required 224x224",
                })

            unique_vals = set(np.unique(mask).tolist())
            invalid_vals = unique_vals - {0, 255}
            if invalid_vals:
                self.errors.append({
                    "sample_id": sample_id,
                    "type": "NON_BINARY_MASK",
                    "detail": f"Mask contains non-binary values: {sorted(list(invalid_vals))[:10]}",
                })

            # 6. Specific mask semantics
            if label == 0:  # REAL -> all zeros
                if not np.all(mask == 0):
                    self.errors.append({
                        "sample_id": sample_id,
                        "type": "REAL_MASK_NOT_ALL_ZERO",
                        "detail": f"REAL sample mask has non-zero pixels (max={mask.max()})",
                    })
            elif forgery_type == "EFS":  # EFS -> all 255
                if not np.all(mask == 255):
                    self.errors.append({
                        "sample_id": sample_id,
                        "type": "EFS_MASK_NOT_ALL_ONE",
                        "detail": f"EFS sample mask has non-255 pixels (min={mask.min()})",
                    })
            elif forgery_type in {"AM", "FS"}:
                # Must have source image
                source_file = self._resolve_path(source_path_str)
                if not source_file or not source_file.exists():
                    self.errors.append({
                        "sample_id": sample_id,
                        "type": "MISSING_SOURCE_FOR_AM_FS",
                        "detail": f"{forgery_type} sample missing source image: {source_path_str}",
                    })
                else:
                    src_img = load_image_rgb(source_file)
                    if src_img is None:
                        self.errors.append({
                            "sample_id": sample_id,
                            "type": "UNREADABLE_SOURCE_IMAGE",
                            "detail": f"Failed to read source image: {source_file}",
                        })
                    elif (src_img.shape[0], src_img.shape[1]) != (224, 224):
                        self.warnings.append({
                            "sample_id": sample_id,
                            "type": "SOURCE_IMAGE_DIMENSION_MISMATCH",
                            "detail": f"Source image dimensions are {src_img.shape[1]}x{src_img.shape[0]}, expected 224x224",
                        })

                # AM/FS mask should not be all 0 (no edit detected) or all 255 (full image changed)
                if np.all(mask == 0):
                    self.warnings.append({
                        "sample_id": sample_id,
                        "type": "AM_FS_MASK_EMPTY",
                        "detail": f"{forgery_type} mask is completely zero (no pixel difference > threshold)",
                    })
                elif np.all(mask == 255):
                    self.warnings.append({
                        "sample_id": sample_id,
                        "type": "AM_FS_MASK_ALL_ONES",
                        "detail": f"{forgery_type} mask is completely 255 (all pixels differ)",
                    })

            # Collect candidates for visualization
            if len(samples_for_viz[forgery_type]) < self.visualize_count:
                samples_for_viz[forgery_type].append({
                    "sample_id": sample_id,
                    "img": img,
                    "mask": mask,
                    "source_path": source_path_str,
                    "target_path": target_path_str,
                    "forgery_type": forgery_type,
                    "generator": generator,
                })

        # Generate sample visualization grids
        if self.visualize_count > 0:
            self._generate_visualizations(samples_for_viz)

        # Build final report
        report = {
            "status": "PASS" if not self.errors else "FAIL",
            "metadata_file": str(self.metadata_csv),
            "dataset_root": str(self.dataset_root),
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "stats": {
                "total_samples": self.stats["total_samples"],
                "real_samples": self.stats["real_samples"],
                "efs_samples": self.stats["efs_samples"],
                "am_samples": self.stats["am_samples"],
                "fs_samples": self.stats["fs_samples"],
                "by_generator": dict(self.stats["by_generator"]),
                "by_architecture": dict(self.stats["by_architecture"]),
                "by_forgery_type": dict(self.stats["by_forgery_type"]),
                "by_split": dict(self.stats["by_split"]),
            },
            "errors": self.errors[:100],  # cap to first 100 errors
            "warnings": self.warnings[:100],
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Verification finished with status: %s", report["status"])
        logger.info("Errors: %d, Warnings: %d", len(self.errors), len(self.warnings))
        logger.info("Full report written to: %s", self.report_path)

        return report

    def _generate_visualizations(self, samples_by_type: Dict[str, List[Dict[str, Any]]]) -> None:
        """Generates side-by-side visualization grids for visual inspection."""
        self.visualize_dir.mkdir(parents=True, exist_ok=True)

        for ftype, sample_list in samples_by_type.items():
            for s in sample_list:
                sample_id = s["sample_id"]
                img = s["img"]
                mask = s["mask"]
                mask_rgb = np.stack([mask, mask, mask], axis=-1)

                source_path = s.get("source_path")
                target_path = s.get("target_path")

                if ftype == "FS" and source_path:
                    # Layout: [SOURCE | TARGET (if present) | FAKE | MASK]
                    src_file = self._resolve_path(source_path)
                    src_img = load_image_rgb(src_file) if src_file else None
                    tgt_file = self._resolve_path(target_path) if target_path else None
                    tgt_img = load_image_rgb(tgt_file) if tgt_file else None

                    panels = []
                    if src_img is not None:
                        panels.append(cv2.resize(src_img, (224, 224), interpolation=cv2.INTER_AREA))
                    if tgt_img is not None:
                        panels.append(cv2.resize(tgt_img, (224, 224), interpolation=cv2.INTER_AREA))
                    panels.extend([img, mask_rgb])
                    grid = np.hstack(panels)

                elif ftype == "AM" and source_path:
                    # Layout: [SOURCE | FAKE | MASK]
                    src_file = self._resolve_path(source_path)
                    src_img = load_image_rgb(src_file) if src_file else None
                    if src_img is not None:
                        src_resized = cv2.resize(src_img, (224, 224), interpolation=cv2.INTER_AREA)
                        grid = np.hstack([src_resized, img, mask_rgb])
                    else:
                        grid = np.hstack([img, mask_rgb])

                else:
                    # Layout: [IMAGE | MASK]
                    grid = np.hstack([img, mask_rgb])

                out_name = f"viz_{ftype}_{sample_id}.png"
                save_image_rgb(self.visualize_dir / out_name, grid)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MFVLR Dataset Verification Tool. Enforces paper reproduction protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="MFVLR_Dataset/metadata/all.csv",
        help="Path to metadata all.csv file.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="MFVLR_Dataset",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="MFVLR_Dataset/logs/dataset_report.json",
        help="Output path for verification report JSON.",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=5,
        help="Number of visual inspection grids to generate per forgery type.",
    )
    parser.add_argument(
        "--visualize-dir",
        type=str,
        default="MFVLR_Dataset/logs/verification_samples",
        help="Directory to save visual inspection sample images.",
    )

    args = parser.parse_args()

    metadata_csv = Path(args.metadata)
    dataset_root = Path(args.root)
    report_path = Path(args.report)
    viz_dir = Path(args.visualize_dir)

    try:
        verifier = DatasetVerifier(
            metadata_csv=metadata_csv,
            dataset_root=dataset_root,
            report_path=report_path,
            visualize_count=args.visualize,
            visualize_dir=viz_dir,
        )
        report = verifier.verify()
        return 0 if report["status"] == "PASS" else 1
    except Exception as e:
        logger.error("Verification failed with exception: %s", e, exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
