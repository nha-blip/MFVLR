"""Build and upload resumable TAR shards for StyleGAN3 images on Kaggle.

The archive directory is the canonical dataset representation on Google Drive.
Each archive contains contiguous, individually named PNG files. No recompression is
performed because PNG data is already compressed.

Actions:
  bootstrap     Rebuild the local uploaded-index manifest from Drive shard names.
  watch         Consume generate_efs.py ready markers, package, upload and checkpoint.

Authentication is read from Kaggle Secret RCLONE_CONFIG_B64.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path


OUTPUT_DIR = Path(os.environ.get(
    "STYLEGAN3_OUTPUT_DIR",
    "/kaggle/working/MFVLR_Dataset/images/EFS/StyleGAN3",
))
QUEUE_DIR = Path(os.environ.get(
    "STYLEGAN3_DRIVE_QUEUE", "/kaggle/working/stylegan3_drive_queue"
))
ARCHIVE_DIR = Path(os.environ.get(
    "STYLEGAN3_SHARD_STAGE", "/kaggle/working/stylegan3_shard_stage"
))
MANIFEST_PATH = Path(os.environ.get(
    "STYLEGAN3_DRIVE_MANIFEST", "/kaggle/working/stylegan3_drive_manifest.json"
))
DRIVE_DEST = os.environ.get(
    "STYLEGAN3_SHARD_DEST", "mainDrive:GenFace/StyleGAN3_shards"
)
SHARD_SIZE = int(os.environ.get("STYLEGAN3_SHARD_SIZE", "500"))
POLL_SECONDS = int(os.environ.get("STYLEGAN3_SHARD_POLL_SECONDS", "15"))
FLUSH_SECONDS = int(os.environ.get("STYLEGAN3_SHARD_FLUSH_SECONDS", "120"))
TARGET_COUNT = int(os.environ.get("STYLEGAN3_TARGET_COUNT", "50000"))
GENERATION_START = int(os.environ.get("STYLEGAN3_GENERATION_START", "12022"))
DELETE_LOCAL_AFTER_UPLOAD = os.environ.get(
    "STYLEGAN3_DELETE_LOCAL_AFTER_UPLOAD", "true"
).lower() in {"1", "true", "yes"}

IMAGE_RE = re.compile(r"^stylegan3_(\d+)\.png$", re.IGNORECASE)
ARCHIVE_RE = re.compile(r"^stylegan3_(\d+)-(\d+)\.tar$", re.IGNORECASE)


def configure_rclone() -> None:
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is missing; install it in this Kaggle session first")
    config_path = Path(os.environ.get("RCLONE_CONFIG", "/kaggle/working/rclone.conf"))
    if not config_path.is_file():
        from kaggle_secrets import UserSecretsClient

        encoded = UserSecretsClient().get_secret("RCLONE_CONFIG_B64")
        # Kaggle Secrets may preserve pasted line breaks or spaces. They are not
        # meaningful in Base64, so remove whitespace while remaining strict about
        # all other characters (e.g. accidental quotes or a wrong value).
        encoded = "".join(encoded.split())
        try:
            config_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError(
                "Kaggle Secret RCLONE_CONFIG_B64 is not valid Base64. Recreate it "
                "from the raw rclone.conf bytes; paste only the Base64 output, "
                "without quotes or a 'data:' prefix."
            ) from exc
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_bytes(config_bytes)
        config_path.chmod(0o600)
    os.environ["RCLONE_CONFIG"] = str(config_path)


def rclone(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["rclone", *args], check=True, text=True, capture_output=capture
    )


def parse_image_index(filename: str) -> int | None:
    match = IMAGE_RE.fullmatch(Path(filename).name)
    return int(match.group(1)) if match else None


def load_manifest() -> set[int]:
    if not MANIFEST_PATH.is_file():
        return set()
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {int(value) for value in data.get("uploaded_indices", [])}


def save_manifest(indices: set[int]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "destination": DRIVE_DEST,
        "uploaded_indices": sorted(indices),
        "updated_at_unix": int(time.time()),
    }), encoding="utf-8")
    os.replace(tmp, MANIFEST_PATH)


def bootstrap() -> set[int]:
    """Shard filenames encode exact inclusive index ranges for fast resume."""
    rclone(["mkdir", DRIVE_DEST])
    listing = rclone(
        ["lsf", "--files-only", "--max-depth", "1", DRIVE_DEST], capture=True
    )
    uploaded: set[int] = set()
    for name in listing.stdout.splitlines():
        match = ARCHIVE_RE.fullmatch(name.strip())
        if not match:
            continue
        first, last = map(int, match.groups())
        if first > last or last >= TARGET_COUNT:
            print(f"Ignoring invalid shard name: {name}", flush=True)
            continue
        uploaded.update(range(first, last + 1))
    save_manifest(uploaded)
    print(
        f"Drive shards found: {len(uploaded):,}/{TARGET_COUNT:,} indices; "
        f"manifest={MANIFEST_PATH}", flush=True,
    )
    return uploaded


def archive_and_upload(paths: list[Path], indices: list[int], uploaded: set[int]) -> None:
    if not paths:
        return
    first, last = indices[0], indices[-1]
    # Enforce contiguous names: the remote archive range must describe its contents exactly.
    if indices != list(range(first, last + 1)):
        raise ValueError("Refusing to create a shard with non-contiguous image indices")
    if len(paths) != len(indices):
        raise ValueError("Image/index count mismatch")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_name = f"stylegan3_{first:06d}-{last:06d}.tar"
    final_path = ARCHIVE_DIR / archive_name
    temp_path = ARCHIVE_DIR / (archive_name + ".partial")
    temp_path.unlink(missing_ok=True)
    with tarfile.open(temp_path, mode="w") as archive:
        for image_path in paths:
            archive.add(image_path, arcname=image_path.name, recursive=False)
    os.replace(temp_path, final_path)

    remote_path = f"{DRIVE_DEST.rstrip('/')}/{archive_name}"
    print(f"Uploading {archive_name} ({len(paths)} PNGs)...", flush=True)
    started = time.monotonic()
    rclone([
        "copyto", str(final_path), remote_path,
        "--retries", "8", "--low-level-retries", "20",
    ])
    elapsed = max(time.monotonic() - started, 0.001)

    # The remote TAR is the durable checkpoint. A new session can reconstruct indices
    # from its filename even if the runtime ends before this local manifest is saved.
    uploaded.update(indices)
    save_manifest(uploaded)
    final_path.unlink(missing_ok=True)
    if DELETE_LOCAL_AFTER_UPLOAD:
        for image_path in paths:
            # Existing Kaggle input data is read-only and must never be removed.
            try:
                image_path.resolve().relative_to(OUTPUT_DIR.resolve())
            except ValueError:
                continue
            image_path.unlink(missing_ok=True)
    print(
        f"Uploaded {len(paths)} images in {elapsed:.1f}s "
        f"({len(paths) / elapsed:.2f} images/s); durable total={len(uploaded):,}.",
        flush=True,
    )


def marker_index(marker: Path) -> tuple[int, str] | None:
    try:
        filename = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    index = parse_image_index(filename)
    return (index, filename) if index is not None else None


def ready_runs(uploaded: set[int]) -> list[list[tuple[int, Path, Path]]]:
    rows: list[tuple[int, Path, Path]] = []
    for marker in QUEUE_DIR.glob("stylegan3_*.ready"):
        parsed = marker_index(marker)
        if parsed is None:
            continue
        index, filename = parsed
        if index in uploaded:
            marker.unlink(missing_ok=True)
            continue
        image_path = OUTPUT_DIR / filename
        if not image_path.is_file() or image_path.stat().st_size == 0:
            continue
        rows.append((index, marker, image_path))
    rows.sort(key=lambda row: row[0])

    runs: list[list[tuple[int, Path, Path]]] = []
    for row in rows:
        if not runs or row[0] != runs[-1][-1][0] + 1:
            runs.append([row])
        else:
            runs[-1].append(row)
    return runs


def watch() -> None:
    uploaded = load_manifest()
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Watching {QUEUE_DIR}; destination={DRIVE_DEST}; shard={SHARD_SIZE}; "
        f"resume={len(uploaded.intersection(range(GENERATION_START, TARGET_COUNT))):,}/"
        f"{TARGET_COUNT - GENERATION_START:,} new indices.", flush=True,
    )
    last_idle_log = 0.0
    expected_new = set(range(GENERATION_START, TARGET_COUNT))
    while not expected_new.issubset(uploaded):
        try:
            runs = ready_runs(uploaded)
            selected: list[tuple[int, Path, Path]] | None = None
            now = time.time()
            for run in runs:
                if len(run) >= SHARD_SIZE:
                    selected = run[:SHARD_SIZE]
                    break
                oldest = min(item[1].stat().st_mtime for item in run)
                if now - oldest >= FLUSH_SECONDS:
                    selected = run
                    break
            if selected:
                indices = [item[0] for item in selected]
                paths = [item[2] for item in selected]
                archive_and_upload(paths, indices, uploaded)
                for _, marker, _ in selected:
                    marker.unlink(missing_ok=True)
            elif time.monotonic() - last_idle_log >= 120:
                queued = sum(len(run) for run in runs)
                durable_new = len(uploaded.intersection(expected_new))
                print(
                    f"Waiting: queued={queued}, durable new={durable_new:,}/"
                    f"{len(expected_new):,}.", flush=True,
                )
                last_idle_log = time.monotonic()
        except Exception as exc:
            print(f"Shard upload error: {exc!r}; queue retained for retry.", flush=True)
        time.sleep(POLL_SECONDS)
    print(
        f"All new indices {GENERATION_START:,}..{TARGET_COUNT - 1:,} are durable "
        "as TAR shards on Drive.", flush=True,
    )


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
