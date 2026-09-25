# Kaggle LatDiff → Google Drive TAR-shard pipeline

Pipeline tương tự StyleGAN3: sinh ảnh thiếu trên 2 GPU Kaggle, lưu PNG tạm vào
`/kaggle/working`, gom 500 ảnh liên tiếp thành TAR rồi upload bằng rclone. 20.536
ảnh cũ đang ở Drive khác nên không tải chúng xuống Kaggle; script bắt đầu sinh từ
index 20.536 và đưa các shard mới vào `mainDrive:GenFace/LatDiff_shards`.

> Trước khi chạy, kiểm tra file cuối của tập cũ. Các bước dưới đây giả định tập cũ
> có tên liên tục `latdiff_000000.png` … `latdiff_020535.png`. Nếu index bắt đầu
> khác 0 hoặc tên khác, cần đổi `LATDIFF_GENERATION_START` và các dải worker.

## 1. Chuẩn bị notebook

Đưa source MFVLR và hai file `generate_efs.py`,
`scripts/drive_latdiff_shard_uploader.py` vào `/kaggle/working/MFVLR`. Bật Internet
trong Notebook Settings. Thêm Kaggle Secret `RCLONE_CONFIG_B64` chứa Base64 của
`rclone.conf` (remote Google Drive tên `mainDrive`). Cài rclone nếu session chưa có:

```bash
!curl https://rclone.org/install.sh | sudo bash
!rclone version
```

## 2. Tải checkpoint LatDiff

Trong thư mục project, dùng downloader của repository để tải checkpoint chính và
VQ-f4 first-stage checkpoint:

```bash
%cd /kaggle/working/MFVLR
!python download_data.py --generator LatDiff --checkpoints-dir /kaggle/working/MFVLR/checkpoints
!ls -lh /kaggle/working/MFVLR/checkpoints/EFS/LatDiff/model.ckpt
!ls -lh /kaggle/working/MFVLR/checkpoints/EFS/LatDiff/first_stage_models/vq-f4/model.ckpt
```

## 3. Đặt cấu hình và kiểm tra Drive

```python
import os

os.environ.update({
    "LATDIFF_OUTPUT_DIR": "/kaggle/working/MFVLR_Dataset/images/EFS/LatDiff",
    "LATDIFF_DRIVE_QUEUE": "/kaggle/working/latdiff_drive_queue",
    "LATDIFF_SHARD_STAGE": "/kaggle/working/latdiff_shard_stage",
    "LATDIFF_DRIVE_MANIFEST": "/kaggle/working/latdiff_drive_manifest.json",
    "LATDIFF_SHARD_DEST": "mainDrive:GenFace/LatDiff_shards",
    "LATDIFF_SHARD_SIZE": "500",
    "LATDIFF_TARGET_COUNT": "60000",
    "LATDIFF_GENERATION_START": "20536",
    "LATDIFF_DELETE_LOCAL_AFTER_UPLOAD": "true",
})
```

```bash
%cd /kaggle/working/MFVLR
!python scripts/drive_latdiff_shard_uploader.py bootstrap
```

Bootstrap chỉ dò TAR trong thư mục đích LatDiff mới; nó không nhìn thấy hay sửa
20.536 ảnh đang nằm trên Drive khác. Không đặt đích vào `StyleGAN3_shards`.

## 4. Clone Latent Diffusion một lần

Chạy trước khi khởi chạy worker để hai GPU không đua nhau clone cùng thư mục:

```python
import subprocess
from pathlib import Path

repo = Path("/kaggle/working/MFVLR/external/latent-diffusion")
if (repo / "ldm").is_dir():
    print("Latent Diffusion source is ready")
else:
    assert not repo.exists(), f"Checkout dở dang: {repo}; kiểm tra/xóa thủ công rồi chạy lại cell này"
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/CompVis/latent-diffusion.git", str(repo)], check=True)
    assert (repo / "ldm").is_dir(), "Clone chưa hoàn chỉnh"
```

## 5. Chạy uploader và 2 GPU

20.536 ảnh đã có; còn **39.464 ảnh**. Hai range dưới đây không chồng lấn:
GPU 0 sinh `[20536,40268)`, GPU 1 sinh `[40268,60000)`. LatDiff chạy 50 DDIM
steps mặc định; để giảm thời gian có thể thử `--steps 25`, nhưng ảnh sẽ khác cấu
hình tham số mặc định. Batch size 1 an toàn hơn trên GPU Kaggle.

```python
import os
import sys
import time
import subprocess
from pathlib import Path

project = Path("/kaggle/working/MFVLR")
checkpoint = project / "checkpoints/EFS/LatDiff/model.ckpt"
out_dir = Path(os.environ["LATDIFF_OUTPUT_DIR"])
log_dir = Path("/kaggle/working/logs")
log_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
assert (project / "generate_efs.py").is_file()
assert checkpoint.is_file(), f"Checkpoint missing: {checkpoint}"

common = [
    "--generator", "LatDiff", "--batch-size", "1", "--steps", "50", "--fp16",
    "--checkpoint", str(checkpoint), "--out-dir", str(out_dir),
    "--drive-manifest", os.environ["LATDIFF_DRIVE_MANIFEST"],
    "--drive-queue-dir", os.environ["LATDIFF_DRIVE_QUEUE"],
]
specs = [
    ("uploader", [sys.executable, "-u", "scripts/drive_latdiff_shard_uploader.py", "watch"], None),
    ("gpu0", [sys.executable, "-u", "generate_efs.py", *common,
              "--start-index", "20536", "--end-index", "40268"], "0"),
    ("gpu1", [sys.executable, "-u", "generate_efs.py", *common,
              "--start-index", "40268", "--end-index", "60000"], "1"),
]

processes, logs = {}, {}
for name, command, gpu in specs:
    child_env = os.environ.copy()
    if gpu is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = gpu
    logs[name] = (log_dir / f"latdiff_{name}.log").open("w", encoding="utf-8")
    processes[name] = subprocess.Popen(command, cwd=project, env=child_env,
                                       stdout=logs[name], stderr=subprocess.STDOUT,
                                       start_new_session=True)
    print(f"Started {name}: PID {processes[name].pid}")

try:
    while True:
        status = {name: proc.poll() for name, proc in processes.items()}
        failures = {name: code for name, code in status.items() if code not in (None, 0)}
        if failures:
            raise RuntimeError(f"Process failure(s): {failures}; inspect /kaggle/working/logs")
        if all(code == 0 for code in status.values()):
            print("Both LatDiff workers and uploader finished")
            break
        print(f"Still running: {status}; logs in /kaggle/working/logs")
        time.sleep(60)
except KeyboardInterrupt:
    print("Stopping Kaggle child processes")
    for proc in processes.values():
        if proc.poll() is None:
            proc.terminate()
finally:
    for proc in processes.values():
        if proc.poll() is None:
            proc.terminate()
    for handle in logs.values():
        handle.close()
```

Kiểm tra log:

```bash
!tail -n 30 /kaggle/working/logs/latdiff_uploader.log
!tail -n 20 /kaggle/working/logs/latdiff_gpu0.log
!tail -n 20 /kaggle/working/logs/latdiff_gpu1.log
```

Watcher chỉ đánh dấu hoàn tất sau khi mọi index 20.536–59.999 đã có trong TAR
trên Drive. Khi Kaggle ngắt, giữ nguyên Drive shards; ở session mới chạy lại cấu
hình, `bootstrap`, rồi chạy lại ba process. Shard đã upload sẽ được nhận diện,
còn ảnh chưa upload có thể được tạo lại. Ảnh cũ trên Drive kia không bị xóa hay
di chuyển. Muốn dataset nằm cùng một thư mục vật lý, sau này cần tải/chép ảnh cũ
về thư mục hợp nhất riêng; pipeline này không tự chuyển dữ liệu giữa hai Drive.
