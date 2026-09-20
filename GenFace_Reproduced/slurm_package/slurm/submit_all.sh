#!/bin/bash
# ==============================================================================
# MFVLR / GenFace-Reproduced: Master Staged Submission Script
# Uses the unified run_dataset.slurm file.
# Supports --dry-run and --submit modes.
# ==============================================================================
set -euo pipefail

MODE=${1:---dry-run}

if [ -f "configs/server.env" ]; then
    set -a; source configs/server.env; set +a
fi

PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/output}
MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-4}

echo "================================================================================"
echo "=== MFVLR / GenFace-Reproduced: Production Submission Orchestrator ==="
echo "================================================================================"
echo "Unified SLURM File:    slurm/run_dataset.slurm"
echo "Target Production Pool: 375,000 fake images (8 Generators)"
echo "  StyleGAN3:    50,000 (50 tasks:  --array=0-49%${MAX_CONCURRENT_TASKS})"
echo "  IAFaces:       5,000 (10 tasks:  --array=0-9%${MAX_CONCURRENT_TASKS})"
echo "  LatTrans:     60,000 (60 tasks:  --array=0-59%${MAX_CONCURRENT_TASKS})"
echo "  FaceSwapper:  30,000 (30 tasks:  --array=0-29%${MAX_CONCURRENT_TASKS})"
echo "  DDPM:         50,000 (100 tasks: --array=0-99%${MAX_CONCURRENT_TASKS})"
echo "  LatDiff:      60,000 (120 tasks: --array=0-119%${MAX_CONCURRENT_TASKS})"
echo "  CollDiff:     50,000 (200 tasks: --array=0-199%${MAX_CONCURRENT_TASKS})"
echo "  DiffAE:       70,000 (280 tasks: --array=0-279%${MAX_CONCURRENT_TASKS})"
echo "Total Array Tasks:     850"
echo "Concurrent Throttling: %${MAX_CONCURRENT_TASKS}"
echo "Output Directory:      ${OUTPUT_ROOT}"
echo "================================================================================"

if [ "$MODE" = "--dry-run" ]; then
    echo "[DRY-RUN MODE] The following commands would be executed:"
    echo "  sbatch --array=0-49%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm stylegan3"
    echo "  sbatch --array=0-9%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm iafaces"
    echo "  sbatch --array=0-59%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm lattrans"
    echo "  sbatch --array=0-29%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm faceswapper"
    echo "  sbatch --array=0-99%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm ddpm"
    echo "  sbatch --array=0-119%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm latdiff"
    echo "  sbatch --array=0-199%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm colldiff"
    echo "  sbatch --array=0-279%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm diffae"
    echo "  (Alternative single master array: sbatch --array=0-849%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm all)"
    echo ""
    echo "To actually submit all jobs, run: ./slurm/submit_all.sh --submit"
    exit 0
fi

if [ "$MODE" = "--submit" ]; then
    echo "WARNING: You are about to submit 850 SLURM array tasks (375,000 samples)!"
    read -r -p "Type 'CONFIRM' to submit: " CONFIRMATION
    if [ "$CONFIRMATION" != "CONFIRM" ]; then
        echo "Submission cancelled by user."
        exit 1
    fi

    mkdir -p logs "$OUTPUT_ROOT"
    echo "Submitting StyleGAN3..."
    JID_SG3=$(sbatch --parsable --array=0-49%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm stylegan3)
    echo "Submitting IAFaces..."
    JID_IAFACES=$(sbatch --parsable --array=0-9%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm iafaces)
    echo "Submitting LatTrans..."
    JID_LATTRANS=$(sbatch --parsable --array=0-59%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm lattrans)
    echo "Submitting FaceSwapper..."
    JID_FACESWAPPER=$(sbatch --parsable --array=0-29%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm faceswapper)
    echo "Submitting DDPM..."
    JID_DDPM=$(sbatch --parsable --array=0-99%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm ddpm)
    echo "Submitting LatDiff..."
    JID_LATDIFF=$(sbatch --parsable --array=0-119%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm latdiff)
    echo "Submitting CollDiff..."
    JID_COLLDIFF=$(sbatch --parsable --array=0-199%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm colldiff)
    echo "Submitting DiffAE..."
    JID_DIFFAE=$(sbatch --parsable --array=0-279%${MAX_CONCURRENT_TASKS} slurm/run_dataset.slurm diffae)

    echo "Submitting dependent verification job..."
    sbatch --dependency=afterok:$JID_SG3:$JID_IAFACES:$JID_LATTRANS:$JID_FACESWAPPER:$JID_DDPM:$JID_LATDIFF:$JID_COLLDIFF:$JID_DIFFAE slurm/run_dataset.slurm verify
    echo "All jobs submitted successfully!"
    exit 0
fi

echo "Unknown option: $MODE. Use --dry-run or --submit."