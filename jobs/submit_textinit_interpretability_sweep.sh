#!/usr/bin/env bash
set -euo pipefail

# Submit multiple textinit training runs focused on interpretability
# Varies: tau, topk, STKIM, diversity loss
# Output: checkpoints/phase1_textinit/

# Usage:
#   bash jobs/submit_textinit_interpretability_sweep.sh
#
# Optional: set DRY_RUN=1 to print commands without submitting:
#   DRY_RUN=1 bash jobs/submit_textinit_interpretability_sweep.sh

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

echo "Submitting textinit interpretability sweep (DRY_RUN=$DRY_RUN)..."
echo "Output directory: checkpoints/phase1_textinit/"
echo ""

TEXT_JOB="jobs/train_cbm_stage_1_textinit_interpretable.job"

# Base configuration: interpretability-focused
BASE_TAU=0.18
BASE_TOPK=4
BASE_STK=0.45
BASE_DIV=0.15
BASE_LK=5

# Configuration 1: Baseline interpretable (recommended)
submit "sbatch --output=jobs/outputs/phase1_textinit/textinit_tau0.18_k4_stk0.45_lk5_div0.15_%j.out --export=ALL,TAU=0.18,TOPK=4,STK_MASK_PROB=0.45,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,ATTN_DIV_W=0.15,ATTN_DIV_TOPM=6,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=textinit_tau0.18_k4_stk0.45_div0.15 $TEXT_JOB"

# Configuration 2: Even sharper attention (lower tau)
submit "sbatch --output=jobs/outputs/phase1_textinit/textinit_tau0.15_k4_stk0.45_lk5_div0.15_%j.out --export=ALL,TAU=0.15,TOPK=4,STK_MASK_PROB=0.45,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,ATTN_DIV_W=0.15,ATTN_DIV_TOPM=6,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=textinit_tau0.15_k4_stk0.45_div0.15 $TEXT_JOB"

# Configuration 3: Very focused (lower topk)
submit "sbatch --output=jobs/outputs/phase1_textinit/textinit_tau0.18_k3_stk0.45_lk5_div0.15_%j.out --export=ALL,TAU=0.18,TOPK=3,STK_MASK_PROB=0.45,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,ATTN_DIV_W=0.15,ATTN_DIV_TOPM=5,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=textinit_tau0.18_k3_stk0.45_div0.15 $TEXT_JOB"

# Configuration 4: Stronger STKIM
submit "sbatch --output=jobs/outputs/phase1_textinit/textinit_tau0.18_k4_stk0.50_lk5_div0.15_%j.out --export=ALL,TAU=0.18,TOPK=4,STK_MASK_PROB=0.50,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,ATTN_DIV_W=0.15,ATTN_DIV_TOPM=6,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=textinit_tau0.18_k4_stk0.50_div0.15 $TEXT_JOB"

# Configuration 5: Higher diversity loss
submit "sbatch --output=jobs/outputs/phase1_textinit/textinit_tau0.18_k4_stk0.45_lk5_div0.20_%j.out --export=ALL,TAU=0.18,TOPK=4,STK_MASK_PROB=0.45,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,ATTN_DIV_W=0.20,ATTN_DIV_TOPM=6,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=textinit_tau0.18_k4_stk0.45_div0.20 $TEXT_JOB"

# Configuration 6: Balanced (slightly higher tau, more patches)
submit "sbatch --output=jobs/outputs/phase1_textinit/textinit_tau0.20_k5_stk0.40_lk5_div0.15_%j.out --export=ALL,TAU=0.20,TOPK=5,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,ATTN_DIV_W=0.15,ATTN_DIV_TOPM=7,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=textinit_tau0.20_k5_stk0.40_div0.15 $TEXT_JOB"

# Configuration 7: Very sharp + very focused
submit "sbatch --output=jobs/outputs/phase1_textinit/textinit_tau0.15_k3_stk0.50_lk5_div0.20_%j.out --export=ALL,TAU=0.15,TOPK=3,STK_MASK_PROB=0.50,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,ATTN_DIV_W=0.20,ATTN_DIV_TOPM=5,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=textinit_tau0.15_k3_stk0.50_div0.20 $TEXT_JOB"

# Configuration 8: Moderate interpretability (less aggressive)
submit "sbatch --output=jobs/outputs/phase1_textinit/textinit_tau0.22_k5_stk0.35_lk5_div0.10_%j.out --export=ALL,TAU=0.22,TOPK=5,STK_MASK_PROB=0.35,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,ATTN_DIV_W=0.10,ATTN_DIV_TOPM=7,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=textinit_tau0.22_k5_stk0.35_div0.10 $TEXT_JOB"

# Configuration 9: Sharp + strong diversity
submit "sbatch --output=jobs/outputs/phase1_textinit/textinit_tau0.18_k4_stk0.45_lk5_div0.25_%j.out --export=ALL,TAU=0.18,TOPK=4,STK_MASK_PROB=0.45,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,ATTN_DIV_W=0.25,ATTN_DIV_TOPM=6,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=textinit_tau0.18_k4_stk0.45_div0.25 $TEXT_JOB"

# Configuration 10: Maximum interpretability (very sharp, very focused, strong regularization)
submit "sbatch --output=jobs/outputs/phase1_textinit/textinit_tau0.12_k3_stk0.50_lk5_div0.25_%j.out --export=ALL,TAU=0.12,TOPK=3,STK_MASK_PROB=0.50,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,ATTN_DIV_W=0.25,ATTN_DIV_TOPM=5,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=textinit_tau0.12_k3_stk0.50_div0.25 $TEXT_JOB"

echo ""
echo "============================================================"
echo "All 10 textinit interpretability jobs submitted!"
echo "Output directory: checkpoints/phase1_textinit/"
echo "Check status: squeue -u \$USER"
echo "============================================================"


