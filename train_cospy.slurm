#!/bin/bash
#SBATCH --job-name=mfvlr_l40
#SBATCH --output=logs/mfvlr_%j.out
#SBATCH --error=logs/mfvlr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:l40:1          # Sử dụng 1 GPU NVIDIA L40 (44GB)
#SBATCH --mem=32G
#SBATCH --time=36:00:00

REQUIRED_VRAM=38000

# 1. Chuyển đến thư mục dự án và tạo thư mục logs, checkpoints
cd ~/data/noname/MFVLR
mkdir -p logs checkpoints

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

echo "============================================================"
echo "Bắt đầu huấn luyện MFVLR trên Slurm Job: $SLURM_JOB_ID"
echo "GPU Device: $CUDA_VISIBLE_DEVICES"
echo "Thư mục làm việc: $(pwd)"
echo "Thời gian bắt đầu: $(date)"
echo "============================================================"

# 5. Chạy huấn luyện MFVLR (GPU L40 44GB: batch-size 32, BFloat16, 4 workers)
python train.py \
    --config configs/mfvlr.yaml \
    --epochs 30 \
    --batch-size 32 \
    --precision bf16 \
    --num-workers 4 \
    --device cuda

echo "============================================================"
echo "Huấn luyện hoàn tất lúc: $(date)"
echo "============================================================"