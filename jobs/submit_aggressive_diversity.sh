#!/usr/bin/env bash
set -euo pipefail

# Submit aggressive diversity loss training based on best setup: tau0.25_k6_stk0.40_lk5_div0.6
# This script creates configurations with higher diversity weights (0.8, 1.0, 1.2)

# Usage:
#   bash jobs/submit_aggressive_diversity.sh
#
# Optional: set DRY_RUN=1 to print commands without submitting:
#   DRY_RUN=1 bash jobs/submit_aggressive_diversity.sh

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

echo "============================================================"
echo "Submitting aggressive diversity loss configurations"
echo "Base config: tau0.25_k6_stk0.40_lk5_div0.6 (BEST)"
echo "Testing: div=0.8, div=1.0, div=1.2"
echo "============================================================"
echo ""

SIMPLE_JOB="jobs/train_cbm_stage_1.job"

# Base configuration: tau0.25_k6_stk0.40_lk5_div0.6 (best performing)
BASE_TAU=0.25
BASE_TOPK=6
BASE_STK=0.40
BASE_LK=5

# Configuration 1: More aggressive diversity (0.8)
echo "Submitting: div=0.8 (33% increase from best)"
submit "sbatch --output=jobs/outputs/phase1_simple/phase1_tau0.25_k6_stk0.40_lk5_div0.8_%j.out --export=ALL,TAU=${BASE_TAU},TOPK=${BASE_TOPK},STK_MASK_PROB=${BASE_STK},STK_K_MASK=1,MIX_LOCAL_KERNEL=${BASE_LK},PROJ_TYPE=simple,ATTN_DIV_W=0.8,ATTN_DIV_TOPM=8,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_tau0.25_k6_stk0.40_lk5_div0.8 $SIMPLE_JOB"

# Configuration 2: Very aggressive diversity (1.0)
echo "Submitting: div=1.0 (67% increase from best)"
submit "sbatch --output=jobs/outputs/phase1_simple/phase1_tau0.25_k6_stk0.40_lk5_div1.0_%j.out --export=ALL,TAU=${BASE_TAU},TOPK=${BASE_TOPK},STK_MASK_PROB=${BASE_STK},STK_K_MASK=1,MIX_LOCAL_KERNEL=${BASE_LK},PROJ_TYPE=simple,ATTN_DIV_W=1.0,ATTN_DIV_TOPM=8,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_tau0.25_k6_stk0.40_lk5_div1.0 $SIMPLE_JOB"

# Configuration 3: Extremely aggressive diversity (1.2)
echo "Submitting: div=1.2 (100% increase from best)"
submit "sbatch --output=jobs/outputs/phase1_simple/phase1_tau0.25_k6_stk0.40_lk5_div1.2_%j.out --export=ALL,TAU=${BASE_TAU},TOPK=${BASE_TOPK},STK_MASK_PROB=${BASE_STK},STK_K_MASK=1,MIX_LOCAL_KERNEL=${BASE_LK},PROJ_TYPE=simple,ATTN_DIV_W=1.2,ATTN_DIV_TOPM=8,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_tau0.25_k6_stk0.40_lk5_div1.2 $SIMPLE_JOB"

echo ""
echo "============================================================"
echo "All aggressive diversity jobs submitted!"
echo "Base config: tau0.25_k6_stk0.40_lk5_div0.6 (best: 44.26% val Acc@1)"
echo "Testing diversity weights: 0.8, 1.0, 1.2"
echo "Output directory: checkpoints/phase1/"
echo "Check status: squeue -u \$USER"
echo "============================================================"


