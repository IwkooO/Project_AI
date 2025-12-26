#!/usr/bin/env bash
set -euo pipefail

# Submit multiple stage 1 training runs to find optimal configuration
# Fixed: div=0.5 (or at least 0.3)
# Varies: tau, topk, STKIM, local kernel
# Output: checkpoints/phase1/

# Usage:
#   bash jobs/submit_stage1_optimization_sweep.sh
#
# Optional: set DRY_RUN=1 to print commands without submitting:
#   DRY_RUN=1 bash jobs/submit_stage1_optimization_sweep.sh

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

echo "Submitting stage 1 optimization sweep (DRY_RUN=$DRY_RUN)..."
echo "Fixed: div=0.5 (or at least 0.3)"
echo "Varies: tau, topk, STKIM, local kernel"
echo "Output directory: checkpoints/phase1/"
echo ""

SIMPLE_JOB="jobs/train_cbm_stage_1.job"



# # ============================================================================
# # TOPK=8 experiments (higher topk for more patches)
# # ============================================================================

# # # Configuration 15: TOPK=8 with DIV=0.0 (no diversity loss)
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k8_stk0.40_lk5_div0.0_%j.out --export=ALL,TAU=0.25,TOPK=8,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=simple,ATTN_DIV_W=0.0,ATTN_DIV_TOPM=10,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_simple_tau0.25_k8_stk0.40_lk5_div0.0 $SIMPLE_JOB"

# # Configuration 16: TOPK=8 with DIV=1.0 (aggressive diversity)
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k8_stk0.40_lk5_div1.0_%j.out --export=ALL,TAU=0.25,TOPK=8,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=simple,ATTN_DIV_W=1.0,ATTN_DIV_TOPM=10,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_simple_tau0.25_k8_stk0.40_lk5_div1.0 $SIMPLE_JOB"

# ============================================================================
# Architecture variants: Baseline -> no stkim
# ============================================================================

# # Configuration 17: Baseline no STKIM with best config (K=6, DIV=1.0)
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k6_stk0.00_lk5_div1.0_%j.out --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.0,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=simple,ATTN_DIV_W=1.0,ATTN_DIV_TOPM=8,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_simple_tau0.25_k6_stk0.00_lk5_div1.0 $SIMPLE_JOB"

# # Configuration 18: Baseline no STKIM with K=8, DIV=1.0
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k8_stk0.00_lk5_div1.0_%j.out --export=ALL,TAU=0.25,TOPK=8,STK_MASK_PROB=0.0,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=simple,ATTN_DIV_W=1.0,ATTN_DIV_TOPM=10,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_simple_tau0.25_k8_stk0.00_lk5_div1.0 $SIMPLE_JOB"

# # Configuration 19: Baseline no STKIM with K=6, DIV=0.0 (no diversity)
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k6_stk0.00_lk5_div0.0_%j.out --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.0,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=simple,ATTN_DIV_W=0.0,ATTN_DIV_TOPM=8,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_simple_tau0.25_k6_stk0.00_lk5_div0.0 $SIMPLE_JOB"

# ============================================================================
# Architecture variants: two_stage projection
# ============================================================================

# Configuration 20: two_stage with best config (K=6, DIV=1.0) - ALREADY RUN
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k6_stk0.40_lk5_div1.0_twostage_%j.out --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=two_stage,ATTN_DIV_W=1.0,ATTN_DIV_TOPM=8,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_tau0.25_k6_stk0.40_lk5_div1.0_twostage $SIMPLE_JOB"

# Configuration 21: two_stage with K=8, DIV=1.0 - ALREADY RUN
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k8_stk0.40_lk5_div1.0_twostage_%j.out --export=ALL,TAU=0.25,TOPK=8,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=two_stage,ATTN_DIV_W=1.0,ATTN_DIV_TOPM=10,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_tau0.25_k8_stk0.40_lk5_div1.0_twostage $SIMPLE_JOB"

# Configuration 22: two_stage with K=6, DIV=0.5 (baseline diversity) - ALREADY RUN
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k6_stk0.40_lk5_div0.5_twostage_%j.out --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=two_stage,ATTN_DIV_W=0.5,ATTN_DIV_TOPM=8,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_tau0.25_k6_stk0.40_lk5_div0.5_twostage $SIMPLE_JOB"

# # Configuration 23: two_stage with K=6, DIV=0.0 (no diversity, test if helps with overfitting)
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k6_stk0.40_lk5_div0.0_twostage_%j.out --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=two_stage,ATTN_DIV_W=0.0,ATTN_DIV_TOPM=8,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_tau0.25_k6_stk0.40_lk5_div0.0_twostage $SIMPLE_JOB"

# # Configuration 24: two_stage with K=6, no STKIM, DIV=1.0 (reduce regularization complexity)
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k6_stk0.00_lk5_div1.0_twostage_%j.out --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.0,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=two_stage,ATTN_DIV_W=1.0,ATTN_DIV_TOPM=8,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_tau0.25_k6_stk0.00_lk5_div1.0_twostage $SIMPLE_JOB"

# ============================================================================
# Architecture variants: bottleneck projection
# ============================================================================

# # Configuration 25: bottleneck with best config (K=6, DIV=1.0)
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k6_stk0.40_lk5_div1.0_bottleneck_%j.out --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=bottleneck,ATTN_DIV_W=1.0,ATTN_DIV_TOPM=8,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_tau0.25_k6_stk0.40_lk5_div1.0_bottleneck $SIMPLE_JOB"

# # Configuration 26: bottleneck with K=8, DIV=1.0
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k8_stk0.40_lk5_div1.0_bottleneck_%j.out --export=ALL,TAU=0.25,TOPK=8,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=bottleneck,ATTN_DIV_W=1.0,ATTN_DIV_TOPM=10,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_tau0.25_k8_stk0.40_lk5_div1.0_bottleneck $SIMPLE_JOB"

# # Configuration 27: bottleneck with K=6, DIV=0.5 (baseline diversity)
# submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k6_stk0.40_lk5_div0.5_bottleneck_%j.out --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=bottleneck,ATTN_DIV_W=0.5,ATTN_DIV_TOPM=8,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_tau0.25_k6_stk0.40_lk5_div0.5_bottleneck $SIMPLE_JOB"


# ============================================================================
# Weird experiments
# ============================================================================

# # Configuration 15: TOPK=2 with DIV=1.0 (test if helps with overfitting)
submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k2_stk0.40_lk5_div1.0_%j.out --export=ALL,TAU=0.25,TOPK=2,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=simple,ATTN_DIV_W=1.0,ATTN_DIV_TOPM=10,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_simple_tau0.25_k2_stk0.40_lk5_div1.0 $SIMPLE_JOB"

# Configuration 16: TOPK=6 with DIV=10.0 (aggressive diversity)
submit "sbatch --output=jobs/outputs/phase1_simple_moredata/phase1_tau0.25_k6_stk0.40_lk5_div10.0_%j.out --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=5,PROJ_TYPE=simple,ATTN_DIV_W=10.0,ATTN_DIV_TOPM=10,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=phase1_simple_tau0.25_k6_stk0.40_lk5_div10.0 $SIMPLE_JOB"


echo ""
echo "============================================================"
echo "Stage 1 optimization jobs submitted!"
echo "TOPK=8 experiments: div=0.0, div=1.0"
echo "Baseline no STKIM: K=6/8 with div=0.0/1.0"
echo "Architecture variants: two_stage (5 configs), bottleneck (3 configs)"
echo "Note: two_stage uses higher dropout (0.4) and weight decay (0.08) to prevent overfitting"
echo "Output directory: checkpoints/phase1_architectures/"
echo "Check status: squeue -u \$USER"
echo "============================================================"

