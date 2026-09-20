#!/bin/bash
#SBATCH --job-name=mfvlr_gen_shard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/slurm_gen_%A_%a.out
#SBATCH --error=logs/slurm_gen_%A_%a.err

# MFVLR Dataset Generation Array Job Template
# Usage:
#   sbatch scripts/slurm/generate_shards.slurm EFS DDPM 1000
#   sbatch scripts/slurm/generate_shards.slurm AM DiffAE 1000
#   sbatch scripts/slurm/generate_shards.slurm FS DiffFace 1000

TYPE=$1        # EFS, AM, or FS
GENERATOR=$2   # e.g. DDPM, DiffAE, DiffFace
COUNT=$3       # Total samples to generate across all shards

if [ -z "$TYPE" ] || [ -z "$GENERATOR" ]; then
    echo "Error: Missing arguments. Usage: sbatch generate_shards.slurm <EFS|AM|FS> <GENERATOR> [COUNT]"
    exit 1
fi

COUNT=${COUNT:-1000}
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-8}

echo "=== MFVLR Generation Task ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Type: $TYPE | Generator: $GENERATOR"
echo "Shard: $SHARD_ID / $NUM_SHARDS | Total Count: $COUNT"

case "$TYPE" in
    EFS)
        python generate_efs.py \
            --generator "$GENERATOR" \
            --count "$COUNT" \
            --shard-id "$SHARD_ID" \
            --num-shards "$NUM_SHARDS" \
            --out-dir MFVLR_Dataset/images/EFS
        ;;
    AM)
        python generate_am.py \
            --generator "$GENERATOR" \
            --source-dir MFVLR_Dataset/images/real \
            --dataset-root MFVLR_Dataset \
            --count "$COUNT" \
            --shard-id "$SHARD_ID" \
            --num-shards "$NUM_SHARDS"
        ;;
    FS)
        python generate_fs.py \
            --generator "$GENERATOR" \
            --source-dir MFVLR_Dataset/images/real \
            --dataset-root MFVLR_Dataset \
            --count "$COUNT" \
            --shard-id "$SHARD_ID" \
            --num-shards "$NUM_SHARDS"
        ;;
    *)
        echo "Unknown type: $TYPE. Expected EFS, AM, or FS."
        exit 1
        ;;
esac

echo "Generation completed for shard $SHARD_ID."
