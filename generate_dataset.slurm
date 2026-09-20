#!/bin/bash
#SBATCH --job-name=genface_l40
#SBATCH --output=logs/genface_%j.out
#SBATCH --error=logs/genface_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=mps:l40:2        # Sử dụng 2 GPU NVIDIA L40 (44GB)
#SBATCH --mem=32G
#SBATCH --time=36:00:00

REQUIRED_VRAM=38000

# 1. Chuyển đến thư mục dự án và tạo thư mục logs, checkpoints
cd ~/data/noname/MFVLR
mkdir -p logs checkpoints

DATA_ROOT="data"
CHECKPOINT_ROOT="checkpoints"
OUTPUT_ROOT="output"

# ------------------------------------------------------------------------------
# 2. Kích hoạt môi trường conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate my_env

# Dừng script ngay lập tức nếu bất kỳ lệnh nào bị lỗi
set -e

# 3. Helper kiểm tra GPU L40 còn trống VRAM
unset CUDA_VISIBLE_DEVICES
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
    exit 0
elif [ $EXIT_CODE -eq 11 ]; then
    exit 1
fi
BEST_GPU=$CHECK_OUT
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# 4. Khởi tạo Private MPS Server
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

# ------------------------------------------------------------------------------
# 5. Thực thi sinh dữ liệu
# ------------------------------------------------------------------------------
TARGET=${1:-all}
TARGET=$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')

echo "============================================================"
echo "Bắt đầu thực thi GenFace SLURM Job: ${SLURM_JOB_ID:-LOCAL}"
echo "Mục tiêu:         $TARGET"
echo "Thư mục làm việc: $(pwd)"
echo "Thời gian bắt đầu: $(date)"
echo "============================================================"

case "$TARGET" in
    download)
        echo "=== Bước 1: Tải trọng số các generator (Checkpoints) ==="
        python download_data.py --checkpoints-dir "$CHECKPOINT_ROOT"
        echo "=== Bước 2: Tải dữ liệu ảnh thật (Real Data) ==="
        python datasets/download_dataset.py --data-dir "$DATA_ROOT/DeepFakeFace"
        ;;

    preflight)
        echo "=== Chạy kiểm tra Preflight ==="
        python scripts/preflight_check.py
        ;;

    benchmark)
        echo "=== Chạy thực nghiệm Benchmark ==="
        python scripts/benchmark_a100.py
        ;;

    verify)
        echo "=== Bước 1: Ghép metadata shards ==="
        python scripts/merge_metadata.py --output-root "$OUTPUT_ROOT"
        echo "=== Bước 2: Kiểm tra tính toàn vẹn dataset ==="
        python scripts/verify_production.py --output-root "$OUTPUT_ROOT"
        ;;

    stylegan3)
        python scripts/generate.py --generator StyleGAN3 --start-index 0 --end-index 50000 --config configs/stylegan3.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
        ;;

    iafaces)
        python scripts/generate.py --generator IAFaces --start-index 0 --end-index 5000 --config configs/iafaces.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
        ;;

    lattrans)
        python scripts/generate.py --generator LatTrans --start-index 0 --end-index 60000 --config configs/lattrans.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
        ;;

    faceswapper)
        python scripts/generate.py --generator FaceSwapper --start-index 0 --end-index 30000 --config configs/faceswapper.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
        ;;

    ddpm)
        python scripts/generate.py --generator DDPM --start-index 0 --end-index 50000 --config configs/ddpm.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
        ;;

    latdiff)
        python scripts/generate.py --generator LatDiff --start-index 0 --end-index 60000 --config configs/latdiff.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
        ;;

    colldiff)
        python scripts/generate.py --generator CollDiff --start-index 0 --end-index 50000 --config configs/colldiff.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
        ;;

    diffae)
        python scripts/generate.py --generator DiffAE --start-index 0 --end-index 70000 --config configs/diffae.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
        ;;

    all)
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
        echo "Số lượng GPU khả dụng: $NUM_GPUS"

        if [ "$NUM_GPUS" -ge 2 ]; then
            echo "============================================================"
            echo "CHẾ ĐỘ SONG SONG 2 GPU (Trong cùng 1 SLURM Job)"
            echo "GPU 0: StyleGAN3 (50k) -> LatTrans (60k) -> FaceSwapper (30k) -> DDPM (50k) [Tổng: 190.000]"
            echo "GPU 1: IAFaces (5k)   -> LatDiff (60k)  -> CollDiff (50k)    -> DiffAE (70k) [Tổng: 185.000]"
            echo "============================================================"

            # Nhánh 1: StyleGAN3, LatTrans, FaceSwapper, DDPM
            (
                echo "[Nhánh 1 - PID $$] 1/4: StyleGAN3 (50,000 ảnh)..."
                python scripts/generate.py --generator StyleGAN3 --start-index 0 --end-index 50000 --config configs/stylegan3.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume

                echo "[Nhánh 1 - PID $$] 2/4: LatTrans (60,000 ảnh)..."
                python scripts/generate.py --generator LatTrans --start-index 0 --end-index 60000 --config configs/lattrans.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume

                echo "[Nhánh 1 - PID $$] 3/4: FaceSwapper (30,000 ảnh)..."
                python scripts/generate.py --generator FaceSwapper --start-index 0 --end-index 30000 --config configs/faceswapper.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume

                echo "[Nhánh 1 - PID $$] 4/4: DDPM (50,000 ảnh)..."
                python scripts/generate.py --generator DDPM --start-index 0 --end-index 50000 --config configs/ddpm.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume

                echo "[Nhánh 1] Hoàn tất toàn bộ lúc $(date)!"
            ) > logs/worker_branch1.log 2>&1 &
            PID1=$!

            # Nhánh 2: IAFaces, LatDiff, CollDiff, DiffAE
            (
                echo "[Nhánh 2 - PID $$] 1/4: IAFaces (5,000 ảnh)..."
                python scripts/generate.py --generator IAFaces --start-index 0 --end-index 5000 --config configs/iafaces.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume

                echo "[Nhánh 2 - PID $$] 2/4: LatDiff (60,000 ảnh)..."
                python scripts/generate.py --generator LatDiff --start-index 0 --end-index 60000 --config configs/latdiff.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume

                echo "[Nhánh 2 - PID $$] 3/4: CollDiff (50,000 ảnh)..."
                python scripts/generate.py --generator CollDiff --start-index 0 --end-index 50000 --config configs/colldiff.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume

                echo "[Nhánh 2 - PID $$] 4/4: DiffAE (70,000 ảnh)..."
                python scripts/generate.py --generator DiffAE --start-index 0 --end-index 70000 --config configs/diffae.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume

                echo "[Nhánh 2] Hoàn tất toàn bộ lúc $(date)!"
            ) > logs/worker_branch2.log 2>&1 &
            PID2=$!

            echo "Đang chạy song song 2 nhánh tiến trình (PID: $PID1 và $PID2)..."
            echo "Theo dõi tiến độ: tail -f logs/worker_branch1.log hoặc tail -f logs/worker_branch2.log"

            wait $PID1
            wait $PID2
            echo "Cả 2 nhánh đã hoàn thành toàn bộ 375,000 ảnh!"
        else
            echo "============================================================"
            echo "CHẾ ĐỘ 1 GPU: Chạy tuần tự lần lượt 8 generator"
            echo "============================================================"
            python scripts/generate.py --generator StyleGAN3 --start-index 0 --end-index 50000 --config configs/stylegan3.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
            python scripts/generate.py --generator IAFaces --start-index 0 --end-index 5000 --config configs/iafaces.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
            python scripts/generate.py --generator LatTrans --start-index 0 --end-index 60000 --config configs/lattrans.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
            python scripts/generate.py --generator FaceSwapper --start-index 0 --end-index 30000 --config configs/faceswapper.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
            python scripts/generate.py --generator DDPM --start-index 0 --end-index 50000 --config configs/ddpm.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
            python scripts/generate.py --generator LatDiff --start-index 0 --end-index 60000 --config configs/latdiff.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
            python scripts/generate.py --generator CollDiff --start-index 0 --end-index 50000 --config configs/colldiff.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
            python scripts/generate.py --generator DiffAE --start-index 0 --end-index 70000 --config configs/diffae.yaml --output-root "$OUTPUT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --data-root "$DATA_ROOT" --task-id 0 --resume
        fi

        echo "=== Tự động ghép metadata và kiểm tra toàn bộ dataset ==="
        python scripts/merge_metadata.py --output-root "$OUTPUT_ROOT"
        python scripts/verify_production.py --output-root "$OUTPUT_ROOT"
        ;;

    *)
        echo "Lỗi: Lựa chọn không hợp lệ '$TARGET'."
        echo "Cách dùng: sbatch generate_dataset.slurm [all|stylegan3|iafaces|lattrans|faceswapper|ddpm|latdiff|colldiff|diffae|preflight|verify]"
        exit 1
        ;;
esac

echo "============================================================"
echo "Job hoàn tất lúc: $(date)"
echo "============================================================"
