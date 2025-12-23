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
#   ./jobs/run_ablation_study.sh                    # Back-compat wrapper
#   ./jobs/launchers/run_ablation_study.sh          # Direct
#
#   ./jobs/launchers/run_ablation_study.sh all
#   ./jobs/launchers/run_ablation_study.sh concept
#   ./jobs/launchers/run_ablation_study.sh image
#   ./jobs/launchers/run_ablation_study.sh both
#
#   ./jobs/launchers/run_ablation_study.sh --vanilla all
#   ./jobs/launchers/run_ablation_study.sh --vanilla concept
#   ./jobs/launchers/run_ablation_study.sh --vanilla image
#   ./jobs/launchers/run_ablation_study.sh --vanilla both

cd /scratch-shared/pnair/Project_AI

# Create output directory if it doesn't exist
mkdir -p jobs/outputs
mkdir -p jobs/outputs/stage1 jobs/outputs/stage1/vanilla
mkdir -p jobs/outputs/stage2 jobs/outputs/stage2/vanilla

echo "============================================="
echo "Stage 2 Ablation Study Launcher"
echo "============================================="

VANILLA_MODE=0
if [ "${1:-}" == "--vanilla" ]; then
    VANILLA_MODE=1
    shift
fi

MODE="${1:-all}"

if [ "${VANILLA_MODE}" -eq 1 ]; then
    echo "Vanilla mode enabled:"
    echo "  - Stage 1: vanilla StreetCLIP (no Stage 0 fine-tuning)"
    echo "  - Stage 2: uses Stage 1 vanilla checkpoint for ablations"
    echo ""

    echo "Submitting Stage 1 vanilla job..."
    STAGE1_JID=$(sbatch --parsable jobs/stage1/train_stage1_vanilla_streetclip.job)
    echo "Stage 1 job id: ${STAGE1_JID}"
    echo ""

    if [ "${MODE}" == "" ] || [ "${MODE}" == "all" ]; then
        echo "Submitting ALL ablation experiments (dependent on Stage 1)..."
        echo ""

        echo "[1/3] Submitting concept_only experiment (vanilla Stage 1)..."
        sbatch --dependency=afterok:${STAGE1_JID} jobs/stage2/vanilla_stage1/train_stage2_ablation_concept_only_vanilla_stage1.job

        echo "[2/3] Submitting image_only experiment (vanilla Stage 1)..."
        sbatch --dependency=afterok:${STAGE1_JID} jobs/stage2/vanilla_stage1/train_stage2_ablation_image_only_vanilla_stage1.job

        echo "[3/3] Submitting both experiment (vanilla Stage 1)..."
        sbatch --dependency=afterok:${STAGE1_JID} jobs/stage2/vanilla_stage1/train_stage2_ablation_both_vanilla_stage1.job

    elif [ "${MODE}" == "concept" ] || [ "${MODE}" == "concept_only" ]; then
        echo "Submitting concept_only experiment (vanilla Stage 1)..."
        sbatch --dependency=afterok:${STAGE1_JID} jobs/stage2/vanilla_stage1/train_stage2_ablation_concept_only_vanilla_stage1.job

    elif [ "${MODE}" == "image" ] || [ "${MODE}" == "image_only" ]; then
        echo "Submitting image_only experiment (vanilla Stage 1)..."
        sbatch --dependency=afterok:${STAGE1_JID} jobs/stage2/vanilla_stage1/train_stage2_ablation_image_only_vanilla_stage1.job

    elif [ "${MODE}" == "both" ]; then
        echo "Submitting both experiment (vanilla Stage 1)..."
        sbatch --dependency=afterok:${STAGE1_JID} jobs/stage2/vanilla_stage1/train_stage2_ablation_both_vanilla_stage1.job

    else
        echo "Unknown option: ${MODE}"
        echo "Usage: $0 [--vanilla] [all|concept|image|both]"
        exit 1
    fi

    echo ""
    echo "============================================="
    echo "Jobs submitted! Check status with: squeue -u \$USER"
    echo "Vanilla Stage 1 checkpoint path will be written to:"
    echo "  - jobs/outputs/stage1_vanilla_ckpt_path.txt"
    echo "Results will be saved to:"
    echo "  - results/stage1-prototype/vanilla_streetclip_no_stage0/"
    echo "  - results/stage2_cross_attention_concept_only/vanilla_stage1/"
    echo "  - results/stage2_cross_attention_image_only/vanilla_stage1/"
    echo "  - results/stage2_cross_attention_both/vanilla_stage1/"
    echo "============================================="
    exit 0
fi

# Non-vanilla: submit Stage0 -> Stage1 -> Stage2 ablations (all dependent)
echo "Submitting Stage 0 finetuning job..."
STAGE0_JID=$(sbatch --parsable jobs/stage0/train_stage0_prototype.job)
echo "Stage 0 job id: ${STAGE0_JID}"
echo ""

echo "Submitting Stage 1 finetuned job (depends on Stage 0)..."
STAGE1_JID=$(sbatch --parsable --dependency=afterok:${STAGE0_JID} jobs/stage1/train_stage1_prototype.job)
echo "Stage 1 job id: ${STAGE1_JID}"
echo ""

if [ "${MODE}" == "" ] || [ "${MODE}" == "all" ]; then
    echo "Submitting ALL ablation experiments..."
    echo ""

    echo "[1/3] Submitting concept_only experiment..."
    sbatch --dependency=afterok:${STAGE1_JID} jobs/stage2/ablations/train_stage2_ablation_concept_only.job

    echo "[2/3] Submitting image_only experiment..."
    sbatch --dependency=afterok:${STAGE1_JID} jobs/stage2/ablations/train_stage2_ablation_image_only.job

    echo "[3/3] Submitting both experiment..."
    sbatch --dependency=afterok:${STAGE1_JID} jobs/stage2/ablations/train_stage2_ablation_both.job

elif [ "${MODE}" == "concept" ] || [ "${MODE}" == "concept_only" ]; then
    echo "Submitting concept_only experiment..."
    sbatch --dependency=afterok:${STAGE1_JID} jobs/stage2/ablations/train_stage2_ablation_concept_only.job

elif [ "${MODE}" == "image" ] || [ "${MODE}" == "image_only" ]; then
    echo "Submitting image_only experiment..."
    sbatch --dependency=afterok:${STAGE1_JID} jobs/stage2/ablations/train_stage2_ablation_image_only.job

elif [ "${MODE}" == "both" ]; then
    echo "Submitting both experiment..."
    sbatch --dependency=afterok:${STAGE1_JID} jobs/stage2/ablations/train_stage2_ablation_both.job

else
    echo "Unknown option: ${MODE}"
    echo "Usage: $0 [--vanilla] [all|concept|image|both]"
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


