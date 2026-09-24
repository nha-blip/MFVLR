"""Resume StyleGAN3 image generation on Kaggle with two GPUs and periodic Kaggle backups.

Run this as a Python script from a Kaggle notebook after making the project,
StyleGAN3 source, and checkpoint available under /kaggle/working or /kaggle/input.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import multiprocessing as mp
import queue
from pathlib import Path
from typing import Any


# --------------------------- Kaggle configuration ---------------------------

TARGET_COUNT = 50_000
BACKUP_EVERY_NEW_IMAGES = 500  # At most about 499 completed images between backups.
BACKUP_POLL_SECONDS = 30

# Set to the folder that already contains the 12,022 generated PNGs.
OUTPUT_DIR = Path(os.environ.get(
    "STYLEGAN3_OUTPUT_DIR",
    "/kaggle/working/MFVLR/MFVLR_Dataset/images/AM/StyleGAN3",
))

# This should be the official StyleGAN3 checkout containing legacy.py.
STYLEGAN3_DIR = Path(os.environ.get(
    "STYLEGAN3_DIR", "/kaggle/working/MFVLR/external/stylegan3"
))

# Set this to the Kaggle input path where the verified FFHQ StyleGAN3 pickle is mounted.
CHECKPOINT = Path(os.environ.get(
    "STYLEGAN3_CHECKPOINT",
    "/kaggle/input/stylegan3-checkpoint/stylegan3-r-ffhq-1024x1024.pkl",
))
CHECKPOINT_SHA256 = os.environ.get(
    "STYLEGAN3_CHECKPOINT_SHA256",
    "ffe2233fa0d0329ad9f19b8bf8ea855d12210461c118ff52fbffde8d3a2b4519",
)

# Use the exact Kaggle dataset slug. Set DATASET_EXISTS=true if it already exists.
KAGGLE_DATASET_SLUG = os.environ.get(
    "KAGGLE_DATASET_SLUG", "YOUR_KAGGLE_USERNAME/stylegan3-generated-faces"
)
DATASET_EXISTS = os.environ.get("KAGGLE_DATASET_EXISTS", "false").lower() == "true"

# Specify a license that you have the right to apply. Leave blank to stop before upload.
KAGGLE_DATASET_LICENSE = os.environ.get("KAGGLE_DATASET_LICENSE", "")

# Kaggle Sessions may expose one or two GPUs. Use both when two are available.
MAX_GPUS = 2

IMAGE_RE = re.compile(r"^stylegan3_(\d+)\.png$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_valid_existing_ids(output_dir: Path) -> set[int]:
    """Resume from completed, readable images named stylegan3_<index>.png."""
    from PIL import Image

    valid: set[int] = set()
    if not output_dir.exists():
        return valid
    for path in output_dir.glob("stylegan3_*.png"):
        match = IMAGE_RE.match(path.name)
        if not match or path.stat().st_size == 0:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            valid.add(int(match.group(1)))
        except Exception:
            print(f"Ignoring incomplete/corrupt image: {path}", flush=True)
    return valid


def worker(gpu_id: int, gpu_count: int, indices: list[int], result_queue: Any) -> None:
    """One independent StyleGAN3 process per GPU; each index has a deterministic seed."""
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(STYLEGAN3_DIR))
    import legacy

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    with CHECKPOINT.open("rb") as stream:
        network = legacy.load_network_pkl(stream)
        generator = network["G_ema"].to(device).eval()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    errors: list[tuple[int, str]] = []

    for index in indices:
        final_path = output_dir / f"stylegan3_{index:05d}.png"
        if final_path.exists() and final_path.stat().st_size > 0:
            continue

        tmp_path = output_dir / f".stylegan3_{index:05d}.png.part"
        try:
            # Per-index seed means GPU scheduling/session restarts do not change samples.
            torch.manual_seed(index)
            latent = torch.randn((1, generator.z_dim), device=device)
            with torch.inference_mode():
                tensor = generator(
                    latent, None, truncation_psi=0.7, noise_mode="const"
                )
                tensor = (
                    (tensor.permute(0, 2, 3, 1) * 127.5 + 128)
                    .clamp(0, 255)
                    .to(torch.uint8)[0]
                    .cpu()
                    .numpy()
                )

            image = Image.fromarray(tensor, mode="RGB").resize(
                (224, 224), Image.Resampling.LANCZOS
            )
            image.save(tmp_path, format="PNG", compress_level=1)
            os.replace(tmp_path, final_path)  # Only complete PNGs become visible to backup.
            completed += 1
            if completed % 100 == 0:
                result_queue.put(("progress", gpu_id, completed, index))
        except Exception as exc:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            errors.append((index, repr(exc)))
            result_queue.put(("error", gpu_id, index, repr(exc)))

    result_queue.put(("done", gpu_id, completed, errors))


def make_upload_snapshot(output_dir: Path, stage_dir: Path) -> int:
    """Hard-link completed images into a stable flat snapshot for Kaggle CLI."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in sorted(output_dir.glob("stylegan3_*.png")):
        if not IMAGE_RE.match(source.name) or source.stat().st_size == 0:
            continue
        target = stage_dir / source.name
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        count += 1
    return count


def upload_snapshot(stage_dir: Path, count: int, is_initial: bool) -> None:
    metadata = stage_dir / "dataset-metadata.json"
    metadata.write_text(json.dumps({
        "title": "StyleGAN3 Generated Face Images",
        "id": KAGGLE_DATASET_SLUG,
        "licenses": [{"name": KAGGLE_DATASET_LICENSE}],
    }, indent=2), encoding="utf-8")

    action = "version" if (DATASET_EXISTS or not is_initial) else "create"
    command = [
        "kaggle", "datasets", action,
        "-p", str(stage_dir),
        "-m", f"StyleGAN3 backup: {count} images",
        "-r", "skip",
    ]
    print(f"Uploading {count:,} PNGs with `kaggle datasets {action}`...", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    if shutil.which("kaggle") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "kaggle"],
            check=True,
        )
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"OUTPUT_DIR does not exist: {OUTPUT_DIR}. Point it at the folder with the 12,022 images."
        )
    if not STYLEGAN3_DIR.joinpath("legacy.py").exists():
        raise FileNotFoundError(f"StyleGAN3 legacy.py not found under {STYLEGAN3_DIR}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"StyleGAN3 checkpoint not found: {CHECKPOINT}")
    actual_sha256 = sha256_file(CHECKPOINT)
    if actual_sha256.lower() != CHECKPOINT_SHA256.lower():
        raise ValueError(
            f"Checkpoint SHA256 mismatch: got {actual_sha256}, "
            f"expected {CHECKPOINT_SHA256}. Set STYLEGAN3_CHECKPOINT_SHA256 only "
            "if you intentionally use a different verified checkpoint."
        )
    if "YOUR_KAGGLE_USERNAME" in KAGGLE_DATASET_SLUG:
        raise ValueError("Set KAGGLE_DATASET_SLUG to your actual username/dataset-slug.")
    if not KAGGLE_DATASET_LICENSE:
        raise ValueError(
            "Set KAGGLE_DATASET_LICENSE to the license applicable to these images before upload."
        )

    # Kaggle API credentials should be stored as Kaggle Secrets, not in this file.
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
    except Exception as exc:
        raise RuntimeError(
            "Could not load KAGGLE_USERNAME/KAGGLE_KEY from Kaggle Secrets."
        ) from exc

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU is enabled for this Kaggle session.")

    gpu_count = min(torch.cuda.device_count(), MAX_GPUS)
    print(f"Visible GPUs: {torch.cuda.device_count()}; using {gpu_count}.", flush=True)
    for gpu_id in range(gpu_count):
        print(f"  cuda:{gpu_id}: {torch.cuda.get_device_name(gpu_id)}", flush=True)
    if gpu_count < 2:
        print("Kaggle exposed only one GPU; generation will continue on that GPU.", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = find_valid_existing_ids(OUTPUT_DIR)
    if existing:
        print(
            f"Found {len(existing):,} valid existing PNGs; index range "
            f"{min(existing)}..{max(existing)}.", flush=True
        )
    remaining = [index for index in range(TARGET_COUNT) if index not in existing]
    if not remaining:
        print(f"Already have {TARGET_COUNT:,} valid images.", flush=True)

    # Prepare stable metadata for the first upload.
    stage_dir = Path("/kaggle/working/stylegan3_kaggle_dataset_snapshot")
    initial_count = make_upload_snapshot(OUTPUT_DIR, stage_dir)
    if initial_count:
        # Publish the resumed local set before generating more, creating or updating as configured.
        upload_snapshot(stage_dir, initial_count, is_initial=not DATASET_EXISTS)
        dataset_created = True
    else:
        dataset_created = DATASET_EXISTS

    if not remaining:
        return

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    per_gpu = [remaining[gpu_id::gpu_count] for gpu_id in range(gpu_count)]
    processes = [
        ctx.Process(
            target=worker,
            args=(gpu_id, gpu_count, per_gpu[gpu_id], result_queue),
            name=f"stylegan3-gpu-{gpu_id}",
        )
        for gpu_id in range(gpu_count)
    ]
    for process in processes:
        process.start()

    last_uploaded_count = initial_count if (DATASET_EXISTS or dataset_created) else 0
    done_workers = 0
    done_gpu_ids: set[int] = set()
    dead_worker_since: dict[int, float] = {}
    last_report = time.time()
    try:
        while done_workers < len(processes):
            while True:
                try:
                    message = result_queue.get(timeout=0.2)
                except queue.Empty:
                    break
                if message[0] == "progress":
                    _, gpu_id, worker_completed, last_index = message
                    print(
                        f"GPU {gpu_id}: {worker_completed} new images completed; last id={last_index}",
                        flush=True,
                    )
                elif message[0] == "error":
                    print(f"Generation error: GPU {message[1]}, id={message[2]}: {message[3]}", flush=True)
                elif message[0] == "done":
                    if message[1] not in done_gpu_ids:
                        done_gpu_ids.add(message[1])
                        done_workers += 1
                    print(
                        f"GPU {message[1]} worker finished; generated {message[2]} images; "
                        f"errors={len(message[3])}.", flush=True
                    )

            for gpu_id, process in enumerate(processes):
                if process.exitcode is not None and gpu_id not in done_gpu_ids:
                    dead_worker_since.setdefault(gpu_id, time.time())
                    if time.time() - dead_worker_since[gpu_id] > 5:
                        raise RuntimeError(
                            f"Worker {process.name} exited unexpectedly with "
                            f"code {process.exitcode}; check the worker output above."
                        )

            now = time.time()
            current_count = sum(
                1 for path in OUTPUT_DIR.glob("stylegan3_*.png")
                if IMAGE_RE.match(path.name) and path.stat().st_size > 0
            )
            should_backup = (
                current_count > 0
                and current_count >= last_uploaded_count + BACKUP_EVERY_NEW_IMAGES
            )
            all_finished = all(not process.is_alive() for process in processes)

            if should_backup or (all_finished and current_count > last_uploaded_count):
                snapshot_count = make_upload_snapshot(OUTPUT_DIR, stage_dir)
                upload_snapshot(stage_dir, snapshot_count, is_initial=not dataset_created)
                dataset_created = True
                last_uploaded_count = snapshot_count
                print(f"Backup confirmed at {snapshot_count:,} images.", flush=True)
                last_report = now

            if now - last_report >= 300:
                print(
                    f"Overall progress: {current_count:,}/{TARGET_COUNT:,}; "
                    f"last Kaggle backup: {last_uploaded_count:,}.", flush=True
                )
                last_report = now
            time.sleep(BACKUP_POLL_SECONDS)

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Worker {process.name} exited with code {process.exitcode}")

    except KeyboardInterrupt:
        print("Stop requested. Saving the completed images to Kaggle before exiting...", flush=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        snapshot_count = make_upload_snapshot(OUTPUT_DIR, stage_dir)
        if snapshot_count > 0:
            upload_snapshot(stage_dir, snapshot_count, is_initial=not dataset_created)

    final_ids = find_valid_existing_ids(OUTPUT_DIR)
    print(f"Final valid image count: {len(final_ids):,}/{TARGET_COUNT:,}", flush=True)
    if len(final_ids) < TARGET_COUNT:
        print("Some image indices are still missing; rerun this script to resume.", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
