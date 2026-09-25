"""Create resumable TAR shards of LatDiff PNGs and upload them with rclone.

Kaggle Secret RCLONE_CONFIG_B64 must contain the Base64-encoded rclone.conf.
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

OUTPUT_DIR = Path(os.environ.get("LATDIFF_OUTPUT_DIR", "/kaggle/working/MFVLR_Dataset/images/EFS/LatDiff"))
QUEUE_DIR = Path(os.environ.get("LATDIFF_DRIVE_QUEUE", "/kaggle/working/latdiff_drive_queue"))
ARCHIVE_DIR = Path(os.environ.get("LATDIFF_SHARD_STAGE", "/kaggle/working/latdiff_shard_stage"))
MANIFEST_PATH = Path(os.environ.get("LATDIFF_DRIVE_MANIFEST", "/kaggle/working/latdiff_drive_manifest.json"))
DRIVE_DEST = os.environ.get("LATDIFF_SHARD_DEST", "mainDrive:GenFace/LatDiff_shards")
SHARD_SIZE = int(os.environ.get("LATDIFF_SHARD_SIZE", "500"))
POLL_SECONDS = int(os.environ.get("LATDIFF_SHARD_POLL_SECONDS", "15"))
FLUSH_SECONDS = int(os.environ.get("LATDIFF_SHARD_FLUSH_SECONDS", "120"))
TARGET_COUNT = int(os.environ.get("LATDIFF_TARGET_COUNT", "60000"))
GENERATION_START = int(os.environ.get("LATDIFF_GENERATION_START", "20536"))
DELETE_LOCAL_AFTER_UPLOAD = os.environ.get("LATDIFF_DELETE_LOCAL_AFTER_UPLOAD", "true").lower() in {"1", "true", "yes"}
IMAGE_RE = re.compile(r"^latdiff_(\d+)\.png$", re.IGNORECASE)
ARCHIVE_RE = re.compile(r"^latdiff_(\d+)-(\d+)\.tar$", re.IGNORECASE)


def configure_rclone() -> None:
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is missing; install it in this Kaggle session first")
    config_path = Path(os.environ.get("RCLONE_CONFIG", "/kaggle/working/rclone.conf"))
    if not config_path.is_file():
        from kaggle_secrets import UserSecretsClient
        encoded = "".join(UserSecretsClient().get_secret("RCLONE_CONFIG_B64").split())
        try:
            config_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError("RCLONE_CONFIG_B64 is invalid Base64; encode raw rclone.conf bytes") from exc
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_bytes(config_bytes)
        config_path.chmod(0o600)
    os.environ["RCLONE_CONFIG"] = str(config_path)


def rclone(args: list[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["rclone", *args], check=True, text=True, capture_output=capture)


def parse_index(filename: str) -> int | None:
    match = IMAGE_RE.fullmatch(Path(filename).name)
    return int(match.group(1)) if match else None


def load_manifest() -> set[int]:
    if not MANIFEST_PATH.is_file():
        return set()
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if data.get("destination") not in (None, DRIVE_DEST):
        raise RuntimeError("Manifest destination differs from LATDIFF_SHARD_DEST; use the correct manifest")
    return {int(value) for value in data.get("uploaded_indices", [])}


def save_manifest(indices: set[int]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"destination": DRIVE_DEST, "uploaded_indices": sorted(indices), "updated_at_unix": int(time.time())}), encoding="utf-8")
    os.replace(tmp, MANIFEST_PATH)


def bootstrap() -> set[int]:
    rclone(["mkdir", DRIVE_DEST])
    listing = rclone(["lsf", "--files-only", "--max-depth", "1", DRIVE_DEST], capture=True)
    uploaded: set[int] = set()
    for name in listing.stdout.splitlines():
        match = ARCHIVE_RE.fullmatch(name.strip())
        if match:
            first, last = map(int, match.groups())
            if first <= last and last < TARGET_COUNT:
                uploaded.update(range(first, last + 1))
    save_manifest(uploaded)
    print(f"Drive shards found: {len(uploaded):,}/{TARGET_COUNT:,} indices; manifest={MANIFEST_PATH}", flush=True)
    return uploaded


def archive_upload(paths: list[Path], indices: list[int], uploaded: set[int]) -> None:
    first, last = indices[0], indices[-1]
    if indices != list(range(first, last + 1)) or len(paths) != len(indices):
        raise ValueError("Refusing to create shard with non-contiguous indices")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    name = f"latdiff_{first:06d}-{last:06d}.tar"
    final = ARCHIVE_DIR / name
    partial = ARCHIVE_DIR / (name + ".partial")
    partial.unlink(missing_ok=True)
    with tarfile.open(partial, "w") as archive:
        for path in paths:
            archive.add(path, arcname=path.name, recursive=False)
    os.replace(partial, final)
    print(f"Uploading {name} ({len(paths)} PNGs)...", flush=True)
    rclone(["copyto", str(final), f"{DRIVE_DEST.rstrip('/')}/{name}", "--retries", "8", "--low-level-retries", "20"])
    uploaded.update(indices)
    save_manifest(uploaded)
    final.unlink(missing_ok=True)
    if DELETE_LOCAL_AFTER_UPLOAD:
        for path in paths:
            try:
                path.resolve().relative_to(OUTPUT_DIR.resolve())
            except ValueError:
                continue
            path.unlink(missing_ok=True)
    print(f"Uploaded {len(paths)}; durable total={len(uploaded):,}", flush=True)


def ready_runs(uploaded: set[int]) -> list[list[tuple[int, Path, Path]]]:
    rows = []
    for marker in QUEUE_DIR.glob("latdiff_*.ready"):
        try:
            filename = marker.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        index = parse_index(filename)
        if index is None:
            continue
        if index in uploaded:
            marker.unlink(missing_ok=True)
            continue
        image = OUTPUT_DIR / filename
        if image.is_file() and image.stat().st_size:
            rows.append((index, marker, image))
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
    expected = set(range(GENERATION_START, TARGET_COUNT))
    print(f"Watching LatDiff queue; destination={DRIVE_DEST}; pending durable={len(uploaded & expected):,}/{len(expected):,}", flush=True)
    last_log = 0.0
    while not expected.issubset(uploaded):
        try:
            runs = ready_runs(uploaded)
            selected = None
            now = time.time()
            for run in runs:
                if len(run) >= SHARD_SIZE:
                    selected = run[:SHARD_SIZE]
                    break
                if now - min(item[1].stat().st_mtime for item in run) >= FLUSH_SECONDS:
                    selected = run
                    break
            if selected:
                archive_upload([row[2] for row in selected], [row[0] for row in selected], uploaded)
                for _, marker, _ in selected:
                    marker.unlink(missing_ok=True)
            elif time.monotonic() - last_log >= 120:
                print(f"Waiting: queued={sum(map(len, runs))}; durable={len(uploaded & expected):,}/{len(expected):,}", flush=True)
                last_log = time.monotonic()
        except Exception as exc:
            print(f"Upload error: {exc!r}; queue retained", flush=True)
        time.sleep(POLL_SECONDS)
    print(f"All LatDiff indices {GENERATION_START}..{TARGET_COUNT - 1} are archived on Drive", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bootstrap", "watch"))
    action = parser.parse_args().action
    configure_rclone()
    bootstrap() if action == "bootstrap" else watch()


if __name__ == "__main__":
    main()
