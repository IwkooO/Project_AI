#!/usr/bin/env bash
set -euo pipefail

# Submit all full-attn configurations with diversity weights ×100
# This tests much stronger diversity regularization

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
echo "Submitting full-attn runs with diversity ×100"
echo "============================================================"
echo ""

cd ~/Project_AI
mkdir -p jobs/outputs/phase1_custom
mkdir -p jobs/outputs/phase1_textinit_custom

# # 1. Text-init: div=0.25 → div=25.0
# echo "Submitting: Text-init (div=25.0, topm=16)"
# TAG="textinit_fullattn_tau0.25_k6_stk0.40_lk0_div25_topm16_warm10"
# submit "sbatch \
#   --output=\"jobs/outputs/phase1_textinit_custom/${TAG}_%j.out\" \
#   --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=0,ATTN_DIV_W=25.0,ATTN_DIV_TOPM=16,ATTN_DIV_WARMUP=10,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=\"${TAG}\" \
#   jobs/train_cbm_stage_1_textinit_interpretable.job"

# 2. Two-stage (strong): div=20 → div=2000
echo "Submitting: Two-stage strong (div=500, topm=32)"
TAG="phase1_two_stage_fullattn_tau0.25_k6_stk0.40_lk0_div500_topm32_warm10"
OUTDIR="checkpoints/phase1_architectures/${TAG}"
submit "sbatch \
  --output=\"jobs/outputs/phase1_custom/${TAG}_%j.out\" \
  --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=0,PROJ_TYPE=two_stage,ATTN_DIV_W=2000.0,ATTN_DIV_TOPM=32,ATTN_DIV_WARMUP=10,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=\"${TAG}\",OUTPUT_DIR=\"${OUTDIR}\" \
  jobs/train_cbm_stage_1.job"

# 3. Simple (strong): div=10 → div=1000
echo "Submitting: Simple strong (div=250, topm=32)"
TAG="phase1_simple_fullattn_tau0.25_k6_stk0.40_lk0_div250_topm32_warm10"
OUTDIR="checkpoints/phase1_architectures/${TAG}"
submit "sbatch \
  --output=\"jobs/outputs/phase1_custom/${TAG}_%j.out\" \
  --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=0,PROJ_TYPE=simple,ATTN_DIV_W=1000.0,ATTN_DIV_TOPM=32,ATTN_DIV_WARMUP=10,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=\"${TAG}\",OUTPUT_DIR=\"${OUTDIR}\" \
  jobs/train_cbm_stage_1.job"

# 4. Two-stage (calm): div=5 → div=500
echo "Submitting: Two-stage calm (div=125, topm=16)"
TAG="phase1_two_stage_fullattn_tau0.25_k6_stk0.40_lk0_div125_topm16_warm10"
OUTDIR="checkpoints/phase1_architectures/${TAG}"
submit "sbatch \
  --output=\"jobs/outputs/phase1_custom/${TAG}_%j.out\" \
  --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=0,PROJ_TYPE=two_stage,ATTN_DIV_W=500.0,ATTN_DIV_TOPM=16,ATTN_DIV_WARMUP=10,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=\"${TAG}\",OUTPUT_DIR=\"${OUTDIR}\" \
  jobs/train_cbm_stage_1.job"

# 5. Two-stage (mild): div=1 → div=100
echo "Submitting: Two-stage mild (div=25, topm=16)"
TAG="phase1_two_stage_fullattn_tau0.25_k6_stk0.40_lk0_div25_topm16_warm10"
OUTDIR="checkpoints/phase1_architectures/${TAG}"
submit "sbatch \
  --output=\"jobs/outputs/phase1_custom/${TAG}_%j.out\" \
  --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=0,PROJ_TYPE=two_stage,ATTN_DIV_W=100.0,ATTN_DIV_TOPM=16,ATTN_DIV_WARMUP=10,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=\"${TAG}\",OUTPUT_DIR=\"${OUTDIR}\" \
  jobs/train_cbm_stage_1.job"

# 6. Bottleneck (mild): div=1 → div=100
echo "Submitting: Bottleneck mild (div=25, topm=8)"
TAG="phase1_bottleneck_fullattn_tau0.25_k6_stk0.40_lk0_div25_topm8_warm5"
OUTDIR="checkpoints/phase1_architectures/${TAG}"
submit "sbatch \
  --output=\"jobs/outputs/phase1_custom/${TAG}_%j.out\" \
  --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=0,PROJ_TYPE=bottleneck,ATTN_DIV_W=100.0,ATTN_DIV_TOPM=8,ATTN_DIV_WARMUP=5,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=\"${TAG}\",OUTPUT_DIR=\"${OUTDIR}\" \
  jobs/train_cbm_stage_1.job"

# 7. Bottleneck (strong): div=20 → div=2000
echo "Submitting: Bottleneck strong (div=500, topm=32)"
TAG="phase1_bottleneck_fullattn_tau0.25_k6_stk0.40_lk0_div500_topm32_warm10"
OUTDIR="checkpoints/phase1_architectures/${TAG}"
submit "sbatch \
  --output=\"jobs/outputs/phase1_custom/${TAG}_%j.out\" \
  --export=ALL,TAU=0.25,TOPK=6,STK_MASK_PROB=0.40,STK_K_MASK=1,MIX_LOCAL_KERNEL=0,PROJ_TYPE=bottleneck,ATTN_DIV_W=2000.0,ATTN_DIV_TOPM=32,ATTN_DIV_WARMUP=10,ATTN_DIV_MODE=offdiag,WANDB_RUN_NAME=\"${TAG}\",OUTPUT_DIR=\"${OUTDIR}\" \
  jobs/train_cbm_stage_1.job"

echo ""
echo "============================================================"
echo "All 7 jobs submitted with diversity ×100!"
echo "Check status: squeue -u \$USER"
echo "============================================================"

