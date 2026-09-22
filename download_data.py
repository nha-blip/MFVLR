#!/usr/bin/env python3
"""
download_data.py - MFVLR Dataset Reproduction Checkpoint & Data Downloader

Features:
- Downloads official/pretrained model checkpoints and public datasets.
- Resume capability with HTTP Range headers.
- SHA-256 integrity verification.
- Avoids overwriting existing valid files.
- Reports MANUAL_DOWNLOAD_REQUIRED for sources behind authentication / cloud storage (Baidu, Google Drive).
- Never attempts to bypass authentication, rate limits, or access controls.
- Config-driven from configs/download_sources.yaml.
- CLI with --help and --dry-run.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_data")


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def download_file(
    url: str,
    dest_path: Path,
    expected_sha256: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Downloads a file with resume support and optional SHA-256 verification."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        if expected_sha256:
            current_sha = compute_sha256(dest_path)
            if current_sha.lower() == expected_sha256.lower():
                logger.info("File %s already exists with matching SHA-256. Skipping.", dest_path.name)
                return True
            else:
                logger.warning("File %s exists but SHA-256 mismatch. Re-downloading.", dest_path.name)
        else:
            logger.info("File %s already exists. Skipping (use --force to overwrite).", dest_path.name)
            return True

    if dry_run:
        logger.info("[DRY-RUN] Would download from %s -> %s", url, dest_path)
        return True

    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    downloaded_size = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {"Range": f"bytes={downloaded_size}-"} if downloaded_size > 0 else {}

    logger.info("Downloading %s ...", url)
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        mode = "ab" if downloaded_size > 0 and response.status_code == 206 else "wb"

        if response.status_code not in (200, 206):
            # Fallback if server doesn't support Range
            response = requests.get(url, stream=True, timeout=30)
            mode = "wb"
            downloaded_size = 0

        total_size = int(response.headers.get("content-length", 0)) + downloaded_size

        with open(temp_path, mode) as f, tqdm(
            total=total_size,
            initial=downloaded_size,
            unit="B",
            unit_scale=True,
            desc=dest_path.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Rename part file to final destination
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
        logger.info("Successfully downloaded: %s", dest_path)

        if expected_sha256:
            calc_sha = compute_sha256(dest_path)
            if calc_sha.lower() != expected_sha256.lower():
                logger.error("SHA-256 verification failed for %s! (got %s, expected %s)", dest_path, calc_sha, expected_sha256)
                return False
            logger.info("SHA-256 verification passed for %s.", dest_path.name)

        return True
    except Exception as e:
        logger.error("Download failed for %s: %s", url, e)
        return False


class CheckpointManager:
    """Manages downloading and manual instructions for generator checkpoints."""

    def __init__(
        self,
        config_path: Path,
        checkpoints_dir: Path = Path("checkpoints"),
    ) -> None:
        self.config_path = config_path.resolve()
        self.checkpoints_dir = checkpoints_dir.resolve()
        self.sources = self._load_sources()

    def _load_sources(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Download sources config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("sources", {})

    def process(
        self,
        target_generator: str = "all",
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Processes downloads or prints manual instructions."""
        status_report: Dict[str, Any] = {
            "downloaded": [],
            "skipped": [],
            "manual_required": [],
            "failed": [],
        }

        generators_to_process = (
            list(self.sources.keys())
            if target_generator.lower() == "all"
            else [target_generator]
        )

        for gen_name in generators_to_process:
            if gen_name not in self.sources:
                logger.error("Generator '%s' not found in %s", gen_name, self.config_path.name)
                status_report["failed"].append({"generator": gen_name, "reason": "Not configured"})
                continue

            info = self.sources[gen_name]
            cat = info.get("category", "OTHER")
            url = info.get("checkpoint_url")
            filename = info.get("filename") or f"{gen_name.lower()}_checkpoint.pth"
            sha256 = info.get("sha256")
            instructions = info.get("manual_instructions", "")

            dest_folder = self.checkpoints_dir / cat / gen_name
            dest_file = dest_folder / filename

            if url:
                success = download_file(
                    url=url,
                    dest_path=dest_file,
                    expected_sha256=sha256,
                    force=force,
                    dry_run=dry_run,
                )
                if success:
                    status_report["downloaded"].append({"generator": gen_name, "path": str(dest_file)})
                    if dest_file.suffix.lower() == ".zip" and not dry_run:
                        import zipfile
                        logger.info("Extracting %s to %s ...", dest_file.name, dest_folder)
                        with zipfile.ZipFile(dest_file, "r") as zf:
                            zf.extractall(dest_folder)
                        logger.info("Extracted %s successfully.", dest_file.name)

                    # For LatDiff, also ensure the first stage VQ-f4 model is present
                    if gen_name == "LatDiff" and not dry_run:
                        vq_dir = dest_folder / "first_stage_models" / "vq-f4"
                        vq_ckpt = vq_dir / "model.ckpt"
                        if not vq_ckpt.exists() or force:
                            vq_url = "https://ommer-lab.com/files/latent-diffusion/vq-f4.zip"
                            vq_zip = vq_dir / "vq-f4.zip"
                            logger.info("Downloading LatDiff first-stage VQ-f4 model...")
                            if download_file(vq_url, vq_zip, force=force):
                                import zipfile
                                with zipfile.ZipFile(vq_zip, "r") as zf:
                                    zf.extractall(vq_dir)
                                logger.info("Extracted VQ-f4 model to %s successfully.", vq_dir)
                else:
                    status_report["failed"].append({"generator": gen_name, "url": url})
            else:
                logger.warning(
                    "\n======================================================\n"
                    "[MANUAL_DOWNLOAD_REQUIRED] Generator: %s\n"
                    "Destination Folder: %s\n"
                    "Repository: %s\n"
                    "Instructions: %s\n"
                    "======================================================",
                    gen_name,
                    dest_folder,
                    info.get("repository", "N/A"),
                    instructions,
                )
                status_report["manual_required"].append({
                    "generator": gen_name,
                    "destination_folder": str(dest_folder),
                    "repository": info.get("repository", ""),
                    "instructions": instructions,
                })

        return status_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MFVLR Checkpoint & Dataset Downloader.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/download_sources.yaml",
        help="Path to download sources YAML configuration.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="all",
        help="Specific generator to download (e.g. DDPM, StyleGAN3) or 'all'.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory to store model checkpoints.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check status and print URLs/destinations without downloading.",
    )

    args = parser.parse_args()

    manager = CheckpointManager(
        config_path=Path(args.config),
        checkpoints_dir=Path(args.checkpoints_dir),
    )

    report = manager.process(
        target_generator=args.generator,
        force=args.force,
        dry_run=args.dry_run,
    )

    logger.info("Checkpoint Processing Summary:")
    logger.info("  Downloaded/Verified: %d", len(report["downloaded"]))
    logger.info("  Manual Required:     %d", len(report["manual_required"]))
    logger.info("  Failed:              %d", len(report["failed"]))

    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
