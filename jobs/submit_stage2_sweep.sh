#!/usr/bin/env bash
set -euo pipefail

# Submit multiple Stage 2 training runs to test different Phase1 checkpoints and modes
# Tests: concept_only, image_only, both
# Output: checkpoints/stage2_*/

# Usage:
#   bash jobs/submit_stage2_sweep.sh
#
# Optional: set DRY_RUN=1 to print commands without submitting:
#   DRY_RUN=1 bash jobs/submit_stage2_sweep.sh

DRY_RUN="${DRY_RUN:-0}"

submit() {
  local cmd="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$cmd"
  else
    eval "$cmd"
    echo "Waiting 2 seconds before next submission..."
    sleep 2
  fi
}

echo "Submitting Stage 2 sweep (DRY_RUN=$DRY_RUN)..."
echo "Testing different Phase1 checkpoints and Stage2 modes"
echo "Output directory: checkpoints/stage2_*/"
echo ""

STAGE2_JOB="jobs/train_stage2.job"

# Best performing Phase1 checkpoints (from reruns)
BASE_CHECKPOINT="checkpoints/reruns/rerun_20251225_184801/SIMPLE_DIM256"

# ============================================================================
# Best performing: K=6, DIV=1.0 (test Acc@1=0.3991)
# ============================================================================

# # Configuration 1: K=6 DIV=1.0, concept_only
# submit "sbatch --output=jobs/outputs/stage2_simple_tau0.25_k6_stk0.40_lk5_div1.0_concept_only_%j.out --export=ALL,PHASE1_CHECKPOINT=${BASE_CHECKPOINT}/PHASE1_SIMPLE_DIM256_TAU0.25_K6_STK0.40_LK5_DIV1.0/phase1/best_phase1.pt,PHASE1_TAU=0.25,PHASE1_TOPK=6,PHASE1_STK=0.40,PHASE1_LK=5,PHASE1_DIV=1.0,PHASE1_PROJ_TYPE=simple,STAGE2_MODE=concept_only,OUTPUT_DIR=checkpoints/stage2_simple_tau0.25_k6_stk0.40_lk5_div1.0_concept_only,WANDB_RUN_NAME=stage2_simple_tau0.25_k6_stk0.40_lk5_div1.0_concept_only $STAGE2_JOB"

# # Configuration 2: K=6 DIV=1.0, image_only
# submit "sbatch --output=jobs/outputs/stage2_simple_tau0.25_k6_stk0.40_lk5_div1.0_image_only_%j.out --export=ALL,PHASE1_CHECKPOINT=${BASE_CHECKPOINT}/PHASE1_SIMPLE_DIM256_TAU0.25_K6_STK0.40_LK5_DIV1.0/phase1/best_phase1.pt,PHASE1_TAU=0.25,PHASE1_TOPK=6,PHASE1_STK=0.40,PHASE1_LK=5,PHASE1_DIV=1.0,PHASE1_PROJ_TYPE=simple,STAGE2_MODE=image_only,OUTPUT_DIR=checkpoints/stage2_simple_tau0.25_k6_stk0.40_lk5_div1.0_image_only,WANDB_RUN_NAME=stage2_simple_tau0.25_k6_stk0.40_lk5_div1.0_image_only $STAGE2_JOB"

# Configuration 3: K=6 DIV=0.5, both
submit "sbatch --output=jobs/outputs/stage2_simple_tau0.25_k6_stk0.40_lk5_div0.5_both_%j.out --export=ALL,PHASE1_CHECKPOINT=${BASE_CHECKPOINT}/PHASE1_SIMPLE_DIM256_TAU0.25_K6_STK0.40_LK5_DIV0.5/phase1/best_phase1.pt,PHASE1_TAU=0.25,PHASE1_TOPK=6,PHASE1_STK=0.40,PHASE1_LK=5,PHASE1_DIV=0.5,PHASE1_PROJ_TYPE=simple,STAGE2_MODE=both,OUTPUT_DIR=checkpoints/stage2_simple_tau0.25_k6_stk0.40_lk5_div0.5_both,WANDB_RUN_NAME=stage2_simple_tau0.25_k6_stk0.40_lk5_div0.5_both $STAGE2_JOB"

# # ============================================================================
# # Third best: K=6, DIV=0.5 (test Acc@1=0.3949)
# # ============================================================================

# # Configuration 7: K=6 DIV=0.5, concept_only
# submit "sbatch --output=jobs/outputs/stage2_simple_tau0.25_k6_stk0.40_lk5_div0.5_concept_only_%j.out --export=ALL,PHASE1_CHECKPOINT=${BASE_CHECKPOINT}/PHASE1_SIMPLE_DIM256_TAU0.25_K6_STK0.40_LK5_DIV0.5/phase1/best_phase1.pt,PHASE1_TAU=0.25,PHASE1_TOPK=6,PHASE1_STK=0.40,PHASE1_LK=5,PHASE1_DIV=0.5,PHASE1_PROJ_TYPE=simple,STAGE2_MODE=concept_only,OUTPUT_DIR=checkpoints/stage2_simple_tau0.25_k6_stk0.40_lk5_div0.5_concept_only,WANDB_RUN_NAME=stage2_simple_tau0.25_k6_stk0.40_lk5_div0.5_concept_only $STAGE2_JOB"

# # Configuration 8: K=6 DIV=0.5, image_only
# submit "sbatch --output=jobs/outputs/stage2_simple_tau0.25_k6_stk0.40_lk5_div0.5_image_only_%j.out --export=ALL,PHASE1_CHECKPOINT=${BASE_CHECKPOINT}/PHASE1_SIMPLE_DIM256_TAU0.25_K6_STK0.40_LK5_DIV0.5/phase1/best_phase1.pt,PHASE1_TAU=0.25,PHASE1_TOPK=6,PHASE1_STK=0.40,PHASE1_LK=5,PHASE1_DIV=0.5,PHASE1_PROJ_TYPE=simple,STAGE2_MODE=image_only,OUTPUT_DIR=checkpoints/stage2_simple_tau0.25_k6_stk0.40_lk5_div0.5_image_only,WANDB_RUN_NAME=stage2_simple_tau0.25_k6_stk0.40_lk5_div0.5_image_only $STAGE2_JOB"

# # Configuration 9: K=6 DIV=0.5, both
# submit "sbatch --output=jobs/outputs/stage2_simple_tau0.25_k6_stk0.40_lk5_div0.5_both_%j.out --export=ALL,PHASE1_CHECKPOINT=${BASE_CHECKPOINT}/PHASE1_SIMPLE_DIM256_TAU0.25_K6_STK0.40_LK5_DIV0.5/phase1/best_phase1.pt,PHASE1_TAU=0.25,PHASE1_TOPK=6,PHASE1_STK=0.40,PHASE1_LK=5,PHASE1_DIV=0.5,PHASE1_PROJ_TYPE=simple,STAGE2_MODE=both,OUTPUT_DIR=checkpoints/stage2_simple_tau0.25_k6_stk0.40_lk5_div0.5_both,WANDB_RUN_NAME=stage2_simple_tau0.25_k6_stk0.40_lk5_div0.5_both $STAGE2_JOB"

echo ""
echo "============================================================"
echo "Stage 2 sweep jobs submitted!"
echo "Total: 12 jobs (4 Phase1 checkpoints × 3 modes)"
echo "Best: K=6 DIV=1.0 (test Acc@1=0.3991)"
echo "Output directory: checkpoints/stage2_*/"
echo "Check status: squeue -u \$USER"
echo "============================================================"

