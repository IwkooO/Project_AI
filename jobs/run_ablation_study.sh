#!/bin/bash
# ===== ABLATION STUDY LAUNCHER =====
# This script submits all three ablation study experiments to the SLURM queue
# 
# Experiments:
#   1. concept_only - Only concept embeddings contribute to location prediction
#   2. image_only   - Only image patches contribute to location prediction  
#   3. both         - Concept + image fusion (full model, default)
#
# Usage:
#   ./jobs/run_ablation_study.sh           # Submit all 3 experiments
#   ./jobs/run_ablation_study.sh concept   # Submit only concept_only
#   ./jobs/run_ablation_study.sh image     # Submit only image_only
#   ./jobs/run_ablation_study.sh both      # Submit only both

cd /scratch-shared/pnair/Project_AI

# Create output directory if it doesn't exist
mkdir -p jobs/outputs

echo "============================================="
echo "Stage 2 Ablation Study Launcher"
echo "============================================="

if [ "$1" == "" ] || [ "$1" == "all" ]; then
    echo "Submitting ALL ablation experiments..."
    echo ""
    
    echo "[1/3] Submitting concept_only experiment..."
    sbatch jobs/train_stage2_ablation_concept_only.job
    
    echo "[2/3] Submitting image_only experiment..."
    sbatch jobs/train_stage2_ablation_image_only.job
    
    echo "[3/3] Submitting both experiment..."
    sbatch jobs/train_stage2_ablation_both.job
    
elif [ "$1" == "concept" ] || [ "$1" == "concept_only" ]; then
    echo "Submitting concept_only experiment..."
    sbatch jobs/train_stage2_ablation_concept_only.job
    
elif [ "$1" == "image" ] || [ "$1" == "image_only" ]; then
    echo "Submitting image_only experiment..."
    sbatch jobs/train_stage2_ablation_image_only.job
    
elif [ "$1" == "both" ]; then
    echo "Submitting both experiment..."
    sbatch jobs/train_stage2_ablation_both.job
    
else
    echo "Unknown option: $1"
    echo "Usage: $0 [all|concept|image|both]"
    exit 1
fi

echo ""
echo "============================================="
echo "Jobs submitted! Check status with: squeue -u \$USER"
echo "Results will be saved to:"
echo "  - results/stage2_cross_attention_concept_only/"
echo "  - results/stage2_cross_attention_image_only/"
echo "  - results/stage2_cross_attention_both/"
echo "============================================="
