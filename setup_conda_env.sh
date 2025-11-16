#!/bin/bash
# Script to create a new conda environment for the project
# This avoids using the corrupted environment from another user

module load 2025
module load Anaconda3/2025.06-1

# Create conda environment in user's own directory
# Use a prefix in the scratch directory to avoid permission issues
ENV_PREFIX="${HOME}/.conda/envs/streetview_pnair"

echo "Creating conda environment at: ${ENV_PREFIX}"

# Create environment with Python 3.11
conda create -p "${ENV_PREFIX}" python=3.11 -y

# Activate the environment
source activate "${ENV_PREFIX}"

# Install essential packages via conda
conda install -y -c conda-forge numpy pandas matplotlib scipy scikit-learn pillow -p "${ENV_PREFIX}"

# Install PyTorch (adjust CUDA version as needed for your cluster)
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -p "${ENV_PREFIX}"

# Install other packages via pip
pip install transformers wandb tqdm

# Install remaining requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "Environment created successfully at: ${ENV_PREFIX}"
echo "To activate: source activate ${ENV_PREFIX}"

