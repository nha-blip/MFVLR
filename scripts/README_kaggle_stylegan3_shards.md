# Kaggle StyleGAN3 → Google Drive TAR-shard pipeline

The Drive shard folder is the canonical copy of the complete dataset. It contains
TAR archives named `stylegan3_<first-index>-<last-index>.tar`; each archive holds
contiguously indexed PNGs. PNGs are already compressed, so TAR packaging is used to
reduce Drive file/API operations rather than to shrink image bytes.

## One-time setup

1. Make the project files available under `/kaggle/working/MFVLR`. The existing
   12,022 images stay in their current Drive folder; they are not copied to Kaggle.
   Add a private code Input containing this project/script if needed.
2. Add Kaggle Secret `RCLONE_CONFIG_B64` with the base64 contents of the local
   `rclone.conf` whose remote is named `mainDrive`. Grant the notebook access.
3. Enable Internet in notebook settings.

Copy the uploader into the project directory. Replace `CODE_INPUT_DIR` with the
actual mounted path containing the script:

```bash
!mkdir -p /kaggle/working/MFVLR/scripts
!cp /kaggle/input/CODE_INPUT_DIR/drive_stylegan3_shard_uploader.py /kaggle/working/MFVLR/scripts/
```

Install rclone:

```bash
!curl https://rclone.org/install.sh | sudo bash
!rclone version
```

## Configure the archive destination

```python
import os

os.environ.update({
    "STYLEGAN3_OUTPUT_DIR": "/kaggle/working/MFVLR_Dataset/images/EFS/StyleGAN3",
    "STYLEGAN3_DRIVE_QUEUE": "/kaggle/working/stylegan3_drive_queue",
    "STYLEGAN3_SHARD_STAGE": "/kaggle/working/stylegan3_shard_stage",
    "STYLEGAN3_DRIVE_MANIFEST": "/kaggle/working/stylegan3_drive_manifest.json",
    "STYLEGAN3_SHARD_DEST": "mainDrive:GenFace/StyleGAN3_shards",
    "STYLEGAN3_SHARD_SIZE": "500",
    "STYLEGAN3_TARGET_COUNT": "50000",
    "STYLEGAN3_GENERATION_START": "12022",
    "STYLEGAN3_DELETE_LOCAL_AFTER_UPLOAD": "true",
})
```

```bash
%cd /kaggle/working/MFVLR
!python scripts/drive_stylegan3_shard_uploader.py bootstrap
```

Bootstrap creates/rebuilds a manifest from any generated-image TARs already in
`mainDrive:GenFace/StyleGAN3_shards`. The 12,022 original PNGs remain untouched in
`mainDrive:GenFace/StyleGAN3`; only indices 12022..49999 are generated and archived.

## Start upload watcher and both GPU workers

Download the official StyleGAN3-R FFHQ checkpoint. Kaggle Internet must be enabled.
The URL is the pretrained model URL listed by the official NVlabs repository.
`wget -c` can resume an interrupted download:

```bash
!mkdir -p /kaggle/working/MFVLR/checkpoints/EFS/StyleGAN3
!wget -c -P /kaggle/working/MFVLR/checkpoints/EFS/StyleGAN3 https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl
!ls -lh /kaggle/working/MFVLR/checkpoints/EFS/StyleGAN3/stylegan3-r-ffhq-1024x1024.pkl
```

Then verify the project/checkpoint paths before launching:

```bash
!test -f /kaggle/working/MFVLR/generate_efs.py && echo 'generator found'
!test -f /kaggle/working/MFVLR/checkpoints/EFS/StyleGAN3/stylegan3-r-ffhq-1024x1024.pkl && echo 'checkpoint found'
!mkdir -p /kaggle/working/logs /kaggle/working/MFVLR_Dataset/images/EFS/StyleGAN3
```

Clone StyleGAN3 once before starting the two workers. This avoids both GPU
processes trying to clone into the same directory at the same time:

```python
import subprocess
from pathlib import Path

repo = Path("/kaggle/working/MFVLR/external/stylegan3")
if (repo / "legacy.py").is_file():
    print("StyleGAN3 source is ready")
else:
    assert not repo.exists(), f"Incomplete checkout already exists: {repo}"
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/NVlabs/stylegan3.git", str(repo)],
        check=True,
    )
    assert (repo / "legacy.py").is_file(), "StyleGAN3 clone is incomplete"
```

Kaggle's IPython shell rejects shell backgrounding with `&`. Instead, run this
Python cell. It starts one uploader and one worker per GPU as child processes, while
the notebook cell stays active and reports their status. The ranges are disjoint
and together cover all remaining indices `[12022, 50000)`:

```python
import os
import sys
import time
import subprocess
from pathlib import Path

project = Path("/kaggle/working/MFVLR")
checkpoint = project / "checkpoints/EFS/StyleGAN3/stylegan3-r-ffhq-1024x1024.pkl"
out_dir = Path("/kaggle/working/MFVLR_Dataset/images/EFS/StyleGAN3")
log_dir = Path("/kaggle/working/logs")
log_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
assert (project / "generate_efs.py").is_file(), "generate_efs.py missing"
assert checkpoint.is_file(), f"Checkpoint missing: {checkpoint}"

common = [
    "--generator", "StyleGAN3", "--batch-size", "1", "--fp16",
    "--checkpoint", str(checkpoint), "--out-dir", str(out_dir),
    "--drive-manifest", "/kaggle/working/stylegan3_drive_manifest.json",
    "--drive-queue-dir", "/kaggle/working/stylegan3_drive_queue",
]
specs = [
    ("uploader", [sys.executable, "-u", "scripts/drive_stylegan3_shard_uploader.py", "watch"], None),
    ("gpu0", [sys.executable, "-u", "generate_efs.py", *common,
              "--start-index", "12022", "--end-index", "31011"], "0"),
    ("gpu1", [sys.executable, "-u", "generate_efs.py", *common,
              "--start-index", "31011", "--end-index", "50000"], "1"),
]

processes, log_handles = {}, {}
for name, command, gpu in specs:
    child_env = os.environ.copy()
    if gpu is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = gpu
    log_handles[name] = (log_dir / f"stylegan3_{name}.log").open("w", encoding="utf-8")
    processes[name] = subprocess.Popen(
        command, cwd=project, env=child_env,
        stdout=log_handles[name], stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"Started {name}: PID {processes[name].pid}", flush=True)

try:
    while True:
        statuses = {name: proc.poll() for name, proc in processes.items()}
        failures = {name: code for name, code in statuses.items()
                    if code not in (None, 0)}
        if failures:
            raise RuntimeError(f"Process failure(s): {failures}; inspect /kaggle/working/logs")
        if all(code == 0 for code in statuses.values()):
            print("All workers and uploader finished successfully.", flush=True)
            break
        print(f"Still running: {statuses}. Logs: /kaggle/working/logs", flush=True)
        for name in processes:
            log_path = log_dir / f"stylegan3_{name}.log"
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if lines:
                    print(f"[{name}] " + "\n".join(lines[-2:]), flush=True)
        time.sleep(60)
except KeyboardInterrupt:
    print("Interrupted: stopping Kaggle child processes.", flush=True)
    for proc in processes.values():
        if proc.poll() is None:
            proc.terminate()
finally:
    for proc in processes.values():
        if proc.poll() is None:
            proc.terminate()
    for handle in log_handles.values():
        handle.close()
```

The generator publishes a `.ready` marker only after an image is completely
written. The watcher archives contiguous ready images in batches of up to 500;
it also uploads a smaller contiguous batch after the flush timeout. On successful
Drive upload it updates the local manifest and removes uploaded local PNGs to limit
Kaggle disk use. Kaggle's `working` disk is temporary; the remote TARs are durable.

After the controller cell finishes, inspect logs if needed:

```bash
!tail -n 30 /kaggle/working/logs/stylegan3_uploader.log
!tail -n 20 /kaggle/working/logs/stylegan3_gpu0.log
!tail -n 20 /kaggle/working/logs/stylegan3_gpu1.log
```

The watcher exits after all 37,978 new indices are represented by uploaded shards.
If Kaggle disconnects, start a fresh session, restore the same environment, run
`bootstrap`, then start the watcher and the same two generation ranges again.
Already archived new ranges are skipped; unarchived generated images are regenerated.
There is no need to download the old flat PNG folder or run `seed-existing`.

## Reading or extracting the dataset

Keep TARs as the canonical storage to avoid creating 50,000 Drive objects. Training
code can stream images from TAR shards. If an image-folder layout is required later,
download the shard folder to local/Colab storage, extract all TARs to a temporary
local directory, verify 50,000 unique filenames, then copy/sync that directory to
Drive. Extracting directly into a Drive mount may be slow because it creates one
Drive file per image and reintroduces the many-small-files bottleneck.
