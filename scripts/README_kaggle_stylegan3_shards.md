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

First verify the generator and checkpoint paths:

```bash
!test -f /kaggle/working/MFVLR/generate_efs.py && echo 'generator found'
!test -f /kaggle/working/MFVLR/checkpoints/EFS/StyleGAN3/stylegan3-r-ffhq-1024x1024.pkl && echo 'checkpoint found'
!mkdir -p /kaggle/working/logs /kaggle/working/MFVLR_Dataset/images/EFS/StyleGAN3
```

Start exactly one uploader, then one generator process per GPU. The ranges are
disjoint and together cover all remaining indices `[12022, 50000)`:

```bash
%cd /kaggle/working/MFVLR
!nohup python scripts/drive_stylegan3_shard_uploader.py watch > /kaggle/working/logs/shard_upload.log 2>&1 < /dev/null &
!nohup env CUDA_VISIBLE_DEVICES=0 python generate_efs.py --generator StyleGAN3 --start-index 12022 --end-index 31011 --batch-size 1 --fp16 --checkpoint /kaggle/working/MFVLR/checkpoints/EFS/StyleGAN3/stylegan3-r-ffhq-1024x1024.pkl --out-dir /kaggle/working/MFVLR_Dataset/images/EFS/StyleGAN3 --drive-manifest /kaggle/working/stylegan3_drive_manifest.json --drive-queue-dir /kaggle/working/stylegan3_drive_queue > /kaggle/working/logs/stylegan3_gpu0.log 2>&1 < /dev/null &
!nohup env CUDA_VISIBLE_DEVICES=1 python generate_efs.py --generator StyleGAN3 --start-index 31011 --end-index 50000 --batch-size 1 --fp16 --checkpoint /kaggle/working/MFVLR/checkpoints/EFS/StyleGAN3/stylegan3-r-ffhq-1024x1024.pkl --out-dir /kaggle/working/MFVLR_Dataset/images/EFS/StyleGAN3 --drive-manifest /kaggle/working/stylegan3_drive_manifest.json --drive-queue-dir /kaggle/working/stylegan3_drive_queue > /kaggle/working/logs/stylegan3_gpu1.log 2>&1 < /dev/null &
```

The generator publishes a `.ready` marker only after an image is completely
written. The watcher archives contiguous ready images in batches of up to 500;
it also uploads a smaller contiguous batch after the flush timeout. On successful
Drive upload it updates the local manifest and removes uploaded local PNGs to limit
Kaggle disk use. Kaggle's `working` disk is temporary; the remote TARs are durable.

Monitor the three logs:

```bash
!tail -n 30 /kaggle/working/logs/shard_upload.log
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
