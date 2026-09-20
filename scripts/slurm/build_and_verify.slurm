#!/bin/bash
#SBATCH --job-name=mfvlr_build_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_build_verify_%j.out
#SBATCH --error=logs/slurm_build_verify_%j.err

# MFVLR Dataset Build, Split, and Verification Job
# Usage:
#   sbatch scripts/slurm/build_and_verify.slurm [PROTOCOL_NAME]

PROTOCOL=${1:-cross_generator}
ROOT=MFVLR_Dataset

echo "=== MFVLR Dataset Build & Verification ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Protocol: $PROTOCOL"

# 1. Build metadata
echo "Step 1: Building metadata..."
python build_metadata.py \
    --root "$ROOT" \
    --config configs/generators.yaml \
    --output "$ROOT/metadata/all.csv"

# 2. Generate ground-truth masks
echo "Step 2: Generating ground-truth masks..."
python generate_masks.py \
    --metadata "$ROOT/metadata/all.csv" \
    --root "$ROOT" \
    --threshold 0.1 \
    --size 224

# 3. Split dataset with anti-leakage verification
echo "Step 3: Splitting dataset (protocol: $PROTOCOL)..."
python split_dataset.py \
    --metadata "$ROOT/metadata/all.csv" \
    --protocol-config configs/protocols.yaml \
    --protocol-name "$PROTOCOL" \
    --output-dir "$ROOT/metadata"

# 4. Verify dataset and generate visual inspection report
echo "Step 4: Running verification..."
python verify_dataset.py \
    --metadata "$ROOT/metadata/all.csv" \
    --root "$ROOT" \
    --report "$ROOT/logs/dataset_report.json" \
    --visualize 5

echo "Pipeline completed successfully."
