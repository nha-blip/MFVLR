"""Periodically publish completed StyleGAN3 PNGs as Kaggle Dataset versions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get(
    "STYLEGAN3_OUTPUT_DIR",
    "/kaggle/working/MFVLR_Dataset/images/EFS/StyleGAN3",
))
STAGE_DIR = Path(os.environ.get(
    "STYLEGAN3_UPLOAD_STAGE",
    "/kaggle/working/stylegan3_upload_stage",
))
STATE_PATH = Path("/kaggle/working/stylegan3_autosave_state.json")
BACKUP_EVERY = int(os.environ.get("STYLEGAN3_BACKUP_EVERY", "500"))
POLL_SECONDS = int(os.environ.get("STYLEGAN3_BACKUP_POLL", "60"))
DATASET_SLUG = os.environ.get("KAGGLE_DATASET_SLUG", "quangnhat6/stylegan3")
DATASET_EXISTS = os.environ.get("KAGGLE_DATASET_EXISTS", "true").lower() == "true"


def load_kaggle_secrets() -> None:
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")


def valid_images() -> list[Path]:
    from PIL import Image

    files = []
    for path in sorted(OUTPUT_DIR.glob("stylegan3_*.png")):
        if path.stat().st_size == 0:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            files.append(path)
        except Exception:
            # A generator may currently be writing this file; pick it up next poll.
            continue
    return files


def snapshot(files: list[Path]) -> int:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    keep = {path.name for path in files}

    for staged in STAGE_DIR.glob("stylegan3_*.png"):
        if staged.name not in keep:
            staged.unlink()

    for source in files:
        target = STAGE_DIR / source.name
        if target.exists():
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return len(files)


def publish(count: int, created: bool) -> None:
    action = "version" if (DATASET_EXISTS or created) else "create"
    metadata = {"id": DATASET_SLUG}
    # License is required when creating a new dataset. For this existing dataset,
    # leave it out so the version update preserves its current metadata.
    if action == "create":
        raise ValueError(
            "This autosaver is configured for the existing dataset. Set "
            "KAGGLE_DATASET_EXISTS=true; creating a new dataset requires a license."
        )
    (STAGE_DIR / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    subprocess.run([
        "kaggle", "datasets", action,
        "-p", str(STAGE_DIR),
        "-m", f"StyleGAN3 autosave: {count} images",
        "-r", "skip",
    ], check=True)


def main() -> None:
    if shutil.which("kaggle") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    load_kaggle_secrets()
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {
        "uploaded_count": 0,
        "created": DATASET_EXISTS,
    }
    print(f"Watching {OUTPUT_DIR}; backup threshold={BACKUP_EVERY} images.", flush=True)

    while True:
        try:
            files = valid_images()
            count = len(files)
            due = count > 0 and (
                count >= int(state["uploaded_count"]) + BACKUP_EVERY
                or int(state["uploaded_count"]) == 0
            )
            if due:
                count = snapshot(files)
                print(f"Publishing {count:,} images to {DATASET_SLUG}...", flush=True)
                publish(count, bool(state["created"]))
                state["uploaded_count"] = count
                state["created"] = True
                STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
                print(f"Backup complete: {count:,} images.", flush=True)
            else:
                print(
                    f"Found {count:,} valid images; last backup="
                    f"{int(state['uploaded_count']):,}.", flush=True
                )
        except Exception as exc:
            print(f"Autosave error: {exc!r}; retrying in {POLL_SECONDS}s.", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
