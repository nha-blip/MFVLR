"""Incrementally upload completed StyleGAN3 PNGs from Kaggle to Google Drive via rclone.

Commands:
  python scripts/drive_stylegan3_uploader.py bootstrap
  python scripts/drive_stylegan3_uploader.py watch

The rclone config is supplied through the Kaggle Secret RCLONE_CONFIG_B64.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path(os.environ.get(
    "STYLEGAN3_OUTPUT_DIR",
    "/kaggle/working/MFVLR_Dataset/images/EFS/StyleGAN3",
))
QUEUE_DIR = Path(os.environ.get(
    "STYLEGAN3_DRIVE_QUEUE",
    "/kaggle/working/stylegan3_drive_queue",
))
STAGE_DIR = Path(os.environ.get(
    "STYLEGAN3_DRIVE_STAGE",
    "/kaggle/working/stylegan3_drive_batch",
))
MANIFEST_PATH = Path(os.environ.get(
    "STYLEGAN3_DRIVE_MANIFEST",
    "/kaggle/working/stylegan3_drive_manifest.json",
))
DRIVE_DEST = os.environ.get(
    "STYLEGAN3_DRIVE_DEST",
    "mainDrive:GenFace/StyleGAN3",
)
DRIVE_MANIFEST_DEST = os.environ.get(
    "STYLEGAN3_DRIVE_MANIFEST_DEST",
    "mainDrive:GenFace/.stylegan3_uploaded_indices.json",
)
BATCH_SIZE = int(os.environ.get("STYLEGAN3_DRIVE_BATCH_SIZE", "500"))
POLL_SECONDS = int(os.environ.get("STYLEGAN3_DRIVE_POLL_SECONDS", "15"))
FLUSH_AFTER_SECONDS = int(os.environ.get("STYLEGAN3_DRIVE_FLUSH_AFTER", "120"))
IMAGE_RE = re.compile(r"^stylegan3_(\d+)\.png$", re.IGNORECASE)


def configure_rclone() -> None:
    """Write the OAuth config from Kaggle Secrets without displaying its contents."""
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is not installed. Install it in the Kaggle session first.")
    config_path = Path(os.environ.get("RCLONE_CONFIG", "/kaggle/working/rclone.conf"))
    if config_path.is_file():
        os.environ["RCLONE_CONFIG"] = str(config_path)
        return
    try:
        from kaggle_secrets import UserSecretsClient
        secret = UserSecretsClient().get_secret("RCLONE_CONFIG_B64")
    except Exception as exc:
        raise RuntimeError(
            "Add the base64-encoded rclone.conf as Kaggle Secret RCLONE_CONFIG_B64."
        ) from exc

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(base64.b64decode(secret))
    config_path.chmod(0o600)
    os.environ["RCLONE_CONFIG"] = str(config_path)


def run_rclone(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    command = ["rclone", *args]
    print("rclone", " ".join(args[:3]), "...", flush=True)
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )


def parse_index(filename: str) -> int | None:
    match = IMAGE_RE.match(Path(filename).name)
    return int(match.group(1)) if match else None


def write_manifest(indices: set[int]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "dataset": DRIVE_DEST,
        "uploaded_indices": sorted(indices),
        "updated_at_unix": int(time.time()),
    }, indent=2), encoding="utf-8")
    os.replace(temporary, MANIFEST_PATH)

    # Keep a small durable resume manifest beside the image folder in Drive.
    remote_temp = STAGE_DIR / "uploaded_indices.json"
    remote_temp.write_bytes(MANIFEST_PATH.read_bytes())
    run_rclone(["copyto", str(remote_temp), DRIVE_MANIFEST_DEST])
    remote_temp.unlink(missing_ok=True)


def bootstrap() -> None:
    """Rebuild the local resume manifest from both Drive files and its sidecar manifest."""
    indices: set[int] = set()

    # Read the sidecar first; it makes restarting independent of downloading image data.
    try:
        sidecar = run_rclone(["cat", DRIVE_MANIFEST_DEST], capture=True)
        data = json.loads(sidecar.stdout)
        indices.update(int(value) for value in data.get("uploaded_indices", []))
    except (subprocess.CalledProcessError, json.JSONDecodeError, TypeError, ValueError):
        print("Drive sidecar is missing or invalid; rebuilding from Drive filenames.", flush=True)

    # Reconcile against actual files in the target directory. This is a one-time listing
    # per Kaggle session; no image bytes are downloaded.
    listing = run_rclone([
        "lsf", "--files-only", "--max-depth", "1", DRIVE_DEST
    ], capture=True)
    for name in listing.stdout.splitlines():
        index = parse_index(name)
        if index is not None:
            indices.add(index)

    write_manifest(indices)
    print(
        f"Resume manifest ready: {len(indices):,} remote StyleGAN3 indices; "
        f"saved at {MANIFEST_PATH}.", flush=True
    )


def load_manifest() -> set[int]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MANIFEST_PATH}; run the bootstrap command before generation."
        )
    data: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {int(value) for value in data.get("uploaded_indices", [])}


def queue_batch(markers: list[Path]) -> tuple[list[Path], set[int]]:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    indices: set[int] = set()

    for marker in markers:
        filename = marker.read_text(encoding="utf-8").strip()
        index = parse_index(filename)
        if index is None:
            print(f"Ignoring malformed queue marker: {marker}", flush=True)
            continue
        source = OUTPUT_DIR / filename
        if not source.is_file() or source.stat().st_size == 0:
            marker.unlink(missing_ok=True)
            continue
        target = STAGE_DIR / filename
        target.unlink(missing_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        staged.append(target)
        indices.add(index)
    return staged, indices


def upload_ready_batch(markers: list[Path], uploaded: set[int]) -> set[int]:
    staged, batch_indices = queue_batch(markers)
    if not staged:
        return uploaded

    run_rclone([
        "copy", str(STAGE_DIR), DRIVE_DEST,
        "--include", "stylegan3_*.png",
        "--ignore-existing",
        "--transfers", "8",
        "--checkers", "8",
        "--retries", "8",
        "--low-level-retries", "20",
    ])

    # Persist confirmed IDs only after rclone reports success.
    uploaded.update(batch_indices)
    write_manifest(uploaded)
    for marker in markers:
        marker.unlink(missing_ok=True)
    for path in staged:
        path.unlink(missing_ok=True)
    print(f"Drive upload confirmed: +{len(batch_indices)}; total={len(uploaded):,}.", flush=True)
    return uploaded


def watch() -> None:
    uploaded = load_manifest()
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Watching {QUEUE_DIR}; destination={DRIVE_DEST}; batch={BATCH_SIZE}.",
        flush=True,
    )

    while True:
        try:
            markers = sorted(QUEUE_DIR.glob("stylegan3_*.ready"))
            if markers:
                oldest_age = time.time() - min(marker.stat().st_mtime for marker in markers)
                if len(markers) >= BATCH_SIZE or oldest_age >= FLUSH_AFTER_SECONDS:
                    batch = markers[:BATCH_SIZE]
                    uploaded = upload_ready_batch(batch, uploaded)
                else:
                    print(f"Queued {len(markers)}/{BATCH_SIZE}; waiting to batch.", flush=True)
        except Exception as exc:
            print(f"Drive upload failed: {exc!r}; queue retained for retry.", flush=True)
        time.sleep(POLL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bootstrap", "watch"))
    args = parser.parse_args()
    configure_rclone()
    if args.action == "bootstrap":
        bootstrap()
    else:
        watch()


if __name__ == "__main__":
    main()
