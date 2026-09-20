#!/bin/bash
# ==============================================================================
# MFVLR / GenFace-Reproduced: Linux Cluster Environment Setup
# Creates and validates the Conda environment for NVIDIA A100 execution
# ==============================================================================
set -euo pipefail

ENV_NAME=${1:-mfvlr_a100}

echo "=== Setting up Conda Environment: ${ENV_NAME} ==="

# Check conda command
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not found in PATH. Please load the Anaconda/Miniconda module or source conda.sh."
    exit 1
fi

# 1. Create environment if it doesn't already exist
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Updating..."
    conda env update -n "${ENV_NAME}" -f env/environment.yml --prune
else
    echo "Creating environment '${ENV_NAME}' from env/environment.yml..."
    conda env create -n "${ENV_NAME}" -f env/environment.yml
fi

# 2. Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "=== Verifying Python and PyTorch Environment ==="

# 3. Verify imports & GPU availability
python -c '
import sys
import torch
import torchvision
import diffusers
import PIL
import numpy as np

print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("WARNING: CUDA is not available to PyTorch in this shell session.")
'

echo "=== Environment Setup Completed Successfully ==="
echo "To activate this environment on the cluster, run:"
echo "    conda activate ${ENV_NAME}"
