#!/usr/bin/env bash
set -euo pipefail

# Rerun script for Phase 1 sweeps with descriptive naming.
# Fixed the race condition and uses explicit full dataset (.bak).

DRY_RUN="${DRY_RUN:-0}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="rerun_${RUN_TIMESTAMP}"

LOG_DIR="jobs/outputs/reruns/${RUN_ROOT}"
CKPT_DIR="checkpoints/reruns/${RUN_ROOT}"

mkdir -p "$LOG_DIR/SIMPLE_DIM256" "$LOG_DIR/TEXTINIT_DIM768"
mkdir -p "$CKPT_DIR/SIMPLE_DIM256" "$CKPT_DIR/TEXTINIT_DIM768"

submit() {
  local cmd="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo -e "\n[DRY-RUN]: $cmd"
  else
    eval "$cmd"
    sleep 2
  fi
}

echo "============================================================"
echo "STARTING PHASE 1 RERUN SWEEP"
echo "RUN TIMESTAMP: $RUN_TIMESTAMP"
echo "LOGS:          $LOG_DIR"
echo "CHECKPOINTS:   $CKPT_DIR"
echo "============================================================"

# 1) phase1_simple (256-dim)
SIMPLE_JOB="jobs/train_cbm_stage_1.job"
echo -e "\n>>> Submitting PHASE1_SIMPLE (256-dim)..."

# Configs: tau topk stk lk div
configs_simple=(
  "0.20 6 0.40 5 0.5"
  "0.25 6 0.40 5 0.3"
  "0.25 6 0.40 5 0.5"
  "0.25 6 0.40 5 0.6"
  "0.25 6 0.40 5 0.8"
  "0.25 6 0.40 5 1.0"
  "0.25 6 0.40 5 1.2"
  "0.25 6 0.50 5 0.5"
  "0.25 7 0.40 5 0.5"
  "0.25 8 0.40 5 0.5"
  "0.30 6 0.40 5 0.5"
)

for c in "${configs_simple[@]}"; do
  read tau k stk lk div <<< "$c"
  # Descriptive ID for filenames and WandB
  NAME="PHASE1_SIMPLE_DIM256_TAU${tau}_K${k}_STK${stk}_LK${lk}_DIV${div}"
  
  submit "sbatch --output=$LOG_DIR/SIMPLE_DIM256/${NAME}_%j.out \
    --export=ALL,TAU=$tau,TOPK=$k,STK_MASK_PROB=$stk,MIX_LOCAL_KERNEL=$lk,ATTN_DIV_W=$div,WANDB_RUN_NAME=${NAME}_${RUN_TIMESTAMP},OUTPUT_DIR=$CKPT_DIR/SIMPLE_DIM256/${NAME} \
    $SIMPLE_JOB"
done

# 2) phase1_textinit (768-dim)
TEXT_JOB="jobs/train_cbm_stage_1_textinit_interpretable.job"
echo -e "\n>>> Submitting PHASE1_TEXTINIT (768-dim)..."

# Configs: tau topk stk lk div
configs_text=(
  "0.18 3 0.45 5 0.15"
  "0.20 5 0.40 5 0.15"
  "0.12 3 0.50 5 0.25"
  "0.18 4 0.45 5 0.15"
  "0.18 4 0.45 5 0.25"
  "0.18 4 0.50 5 0.15"
  "0.15 3 0.50 5 0.20"
)

for c in "${configs_text[@]}"; do
  read tau k stk lk div <<< "$c"
  # Descriptive ID for filenames and WandB
  NAME="PHASE1_TEXTINIT_DIM768_TAU${tau}_K${k}_STK${stk}_LK${lk}_DIV${div}"
  
  submit "sbatch --output=$LOG_DIR/TEXTINIT_DIM768/${NAME}_%j.out \
    --export=ALL,TAU=$tau,TOPK=$k,STK_MASK_PROB=$stk,MIX_LOCAL_KERNEL=$lk,ATTN_DIV_W=$div,WANDB_RUN_NAME=${NAME}_${RUN_TIMESTAMP},OUTPUT_DIR=$CKPT_DIR/TEXTINIT_DIM768/${NAME} \
    $TEXT_JOB"
done

echo -e "\n============================================================"
echo "SUCCESS: All jobs queued with descriptive names."
echo "Check status with: squeue -u $USER"
echo "============================================================"
