#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=jobs/outputs/test.log

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate streetview_pnair
export PYTHONNOUSERSITE=1

python simple-cbm.py