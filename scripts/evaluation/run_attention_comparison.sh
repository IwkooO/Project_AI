#!/usr/bin/env bash
# Submit 6 attention comparison jobs to SLURM queue simultaneously
# Each job compares all 6 checkpoints with different random samples

set -euo pipefail

# Number of samples to visualize per job (can be overridden)
NUM_SAMPLES="${NUM_SAMPLES:-25}"

# Checkpoints to compare (same for all jobs)
# Top 6 models based on test Acc@1 performance:
# 1. textinit_tau0.15_k3_stk0.50_lk5: Acc@1=0.4358, Acc@5=0.7319 (BEST - textinit)
# 2. two_stage_tau0.25_k6_stk0.40_lk5_div0.0: Acc@1=0.4157, Acc@5=0.7122
# 3. two_stage_tau0.25_k6_stk0.0_lk5_div1.0: Acc@1=0.4084, Acc@5=0.7099
# 4. simple_tau0.25_k8_stk0.40_lk5_div0.0: Acc@1=0.4062, Acc@5=0.7115
# 5. simple_tau0.25_k6_stk0.40_lk5_div1.0 (reruns): Acc@1=0.3991, Acc@5=0.7120
# 6. simple_tau0.25_k7_stk0.40_lk5_div0.5 (reruns): Acc@1=0.3980, Acc@5=0.7004

CHECKPOINT_1="${CHECKPOINT_1:-checkpoints/phase1_textinit/cbm_stage_1_textinit_visinit_interpretable_tau0.15_k3_stk0.50_lk5}"
CHECKPOINT_2="${CHECKPOINT_2:-checkpoints/phase1_architectures/cbm_stage_1_two_stage_tau0.25_k6_stk0.40_lk5_div0.0}"
CHECKPOINT_3="${CHECKPOINT_3:-checkpoints/phase1_architectures/cbm_stage_1_two_stage_tau0.25_k6_stk0.0_lk5_div1.0}"
CHECKPOINT_4="${CHECKPOINT_4:-checkpoints/phase1_architectures/cbm_stage_1_simple_tau0.25_k8_stk0.40_lk5_div0.0}"
CHECKPOINT_5="${CHECKPOINT_5:-checkpoints/reruns/rerun_20251225_184801/SIMPLE_DIM256/PHASE1_SIMPLE_DIM256_TAU0.25_K6_STK0.40_LK5_DIV1.0}"
CHECKPOINT_6="${CHECKPOINT_6:-checkpoints/reruns/rerun_20251225_184801/SIMPLE_DIM256/PHASE1_SIMPLE_DIM256_TAU0.25_K7_STK0.40_LK5_DIV0.5}"

echo "============================================================"
echo "Submitting 6 Attention Comparison Jobs to SLURM Queue"
echo "============================================================"
echo "Samples per job: $NUM_SAMPLES"
echo ""
echo "Models to compare (all jobs):"
echo "  1. $CHECKPOINT_1"
echo "  2. $CHECKPOINT_2"
echo "  3. $CHECKPOINT_3"
echo "  4. $CHECKPOINT_4"
echo "  5. $CHECKPOINT_5"
echo "  6. $CHECKPOINT_6"
echo "============================================================"

# Base export variables (same for all jobs)
BASE_EXPORT="ALL"
BASE_EXPORT="${BASE_EXPORT},CHECKPOINT_1=${CHECKPOINT_1}"
BASE_EXPORT="${BASE_EXPORT},CHECKPOINT_2=${CHECKPOINT_2}"
BASE_EXPORT="${BASE_EXPORT},CHECKPOINT_3=${CHECKPOINT_3}"
BASE_EXPORT="${BASE_EXPORT},CHECKPOINT_4=${CHECKPOINT_4}"
BASE_EXPORT="${BASE_EXPORT},CHECKPOINT_5=${CHECKPOINT_5}"
BASE_EXPORT="${BASE_EXPORT},CHECKPOINT_6=${CHECKPOINT_6}"
BASE_EXPORT="${BASE_EXPORT},NUM_SAMPLES=${NUM_SAMPLES}"

# Submit 6 jobs simultaneously
JOB_FILE="jobs/compare_attention_models.job"
JOB_IDS=()

for i in {1..6}; do
    # Each job gets a unique output directory
    EXPORT_VARS="${BASE_EXPORT},OUTPUT_DIR=comparisons/attention_comparison_batch${i}_$(date +%Y%m%d_%H%M%S)"
    
    JOB_ID=$(sbatch --export="$EXPORT_VARS" "$JOB_FILE" | grep -oP '\d+$')
    JOB_IDS+=($JOB_ID)
    echo "  ✓ Job $i queued: ID $JOB_ID"
done

echo ""
echo "============================================================"
echo "All 6 jobs queued successfully!"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Check status: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo "View outputs:"
for i in "${!JOB_IDS[@]}"; do
    echo "  Job $((i+1)): tail -f jobs/outputs/compare_attention_models_${JOB_IDS[$i]}.out"
done
echo "============================================================"

