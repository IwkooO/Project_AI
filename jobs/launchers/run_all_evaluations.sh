#!/bin/bash
# ===== COMPREHENSIVE EVALUATION LAUNCHER =====
# This script runs all evaluation scripts in batch mode and generates final report
#
# Structure:
#   Stage 1: Auto-detects latest checkpoints for both variants
#   Stage 2: Auto-detects latest checkpoints for all ablation modes (both variants each)
#
# Usage:
#   ./jobs/launchers/run_all_evaluations.sh
#
# Environment variables (optional):
#   SPLITS_JSON - Path to splits.json (default: results/stage1-prototype/vanilla_streetclip_no_stage0/2025-12-21_10-48-19/splits.json)
#   CSV_PATH - Path to CSV dataset (default: data/dataset-43k-mapped.csv)
#   DATA_ROOT - Data root directory (default: /scratch-shared/pnair/Project_AI/data)
#   RESULTS_ROOT - Results root for report (default: results)
#   MAX_SAMPLES - Max samples for HF eval (optional, for quick testing)

set -e

cd /scratch-shared/pnair/Project_AI

# Create output directories
mkdir -p jobs/outputs/evaluation

echo "============================================="
echo "Comprehensive Evaluation Launcher (Batch Mode)"
echo "============================================="
echo ""

# Default paths
DEFAULT_SPLITS_JSON="results/stage1-prototype/vanilla_streetclip_no_stage0/2025-12-21_10-48-19/splits.json"
DEFAULT_CSV_PATH="data/dataset-43k-mapped.csv"
DEFAULT_DATA_ROOT="/scratch-shared/pnair/Project_AI/data"
DEFAULT_RESULTS_ROOT="results"

# Use provided or defaults
if [ -z "${SPLITS_JSON:-}" ]; then
  # If user has a latest Stage1 checkpoint path file, prefer its splits.json for consistency.
  if [ -f "jobs/outputs/stage1_ckpt_path.txt" ]; then
    S1_DIR="$(dirname "$(dirname "$(cat jobs/outputs/stage1_ckpt_path.txt)")")"
    SPLITS_JSON="${S1_DIR}/splits.json"
  elif [ -f "jobs/outputs/stage1_vanilla_ckpt_path.txt" ]; then
    S1_DIR="$(dirname "$(dirname "$(cat jobs/outputs/stage1_vanilla_ckpt_path.txt)")")"
    SPLITS_JSON="${S1_DIR}/splits.json"
  else
    SPLITS_JSON="$DEFAULT_SPLITS_JSON"
  fi
else
  SPLITS_JSON="${SPLITS_JSON}"
fi
CSV_PATH="${CSV_PATH:-$DEFAULT_CSV_PATH}"
DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
RESULTS_ROOT="${RESULTS_ROOT:-$DEFAULT_RESULTS_ROOT}"

echo "Configuration:"
echo "  Splits JSON: $SPLITS_JSON"
echo "  CSV Path: $CSV_PATH"
echo "  Data Root: $DATA_ROOT"
echo "  Results Root: $RESULTS_ROOT"
echo ""

# Submit 3 batch evaluation jobs
echo "============================================="
echo "Submitting batch evaluation jobs..."
echo "============================================="

# Job 1: Stage 1 batch evaluation (2 checkpoints)
echo "  Submitting Stage 1 batch evaluation..."
STAGE1_JID=$(sbatch --parsable \
    --export=ALL,SPLITS_JSON="$SPLITS_JSON",CSV_PATH="$CSV_PATH",DATA_ROOT="$DATA_ROOT",RESULTS_ROOT="$RESULTS_ROOT" \
    jobs/evaluation/eval_stage1_test_split.job)
echo "    Job ID: $STAGE1_JID"

# Job 2: Stage 2 test split batch evaluation (6 checkpoints)
echo "  Submitting Stage 2 test split batch evaluation..."
STAGE2_TEST_JID=$(sbatch --parsable \
    --export=ALL,SPLITS_JSON="$SPLITS_JSON",CSV_PATH="$CSV_PATH",DATA_ROOT="$DATA_ROOT",RESULTS_ROOT="$RESULTS_ROOT" \
    jobs/evaluation/eval_stage2_test_split.job)
echo "    Job ID: $STAGE2_TEST_JID"

# Job 3: Stage 2 HF batch evaluation (6 checkpoints)
echo "  Submitting Stage 2 HF batch evaluation..."
STAGE2_HF_JID=$(sbatch --parsable \
    --export=ALL,RESULTS_ROOT="$RESULTS_ROOT"${MAX_SAMPLES:+,MAX_SAMPLES="$MAX_SAMPLES"} \
    jobs/evaluation/eval_hf_geoguessr.job)
echo "    Job ID: $STAGE2_HF_JID"
echo ""

# All evaluation job IDs
ALL_JIDS=("$STAGE1_JID" "$STAGE2_TEST_JID" "$STAGE2_HF_JID")

echo "============================================="
echo "Submitting report generation job..."
echo "============================================="
DEPENDENCY=$(IFS=:; echo "${ALL_JIDS[*]}")
REPORT_JID=$(sbatch --parsable \
    --dependency=afterok:"$DEPENDENCY" \
    --export=ALL,RESULTS_ROOT="$RESULTS_ROOT" \
    jobs/evaluation/build_report.job)
echo "Report generation job ID: $REPORT_JID"
echo ""

echo "============================================="
echo "All jobs submitted!"
echo "============================================="
echo ""
echo "Job Summary:"
echo "  Stage 1 batch evaluation:     $STAGE1_JID (2 checkpoints)"
echo "  Stage 2 test batch evaluation: $STAGE2_TEST_JID (6 checkpoints)"
echo "  Stage 2 HF batch evaluation:  $STAGE2_HF_JID (6 checkpoints)"
echo "  Report generation:             $REPORT_JID (depends on all above)"
echo ""
echo "Total jobs: 4 (3 evaluation + 1 report)"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "Check outputs in:"
echo "  jobs/outputs/evaluation/"
echo ""
echo "Consolidated CSVs will be in:"
echo "  $RESULTS_ROOT/evals/stage1_test_consolidated.csv"
echo "  $RESULTS_ROOT/evals/stage2_test_consolidated.csv"
echo "  $RESULTS_ROOT/evals/stage2_hf_consolidated.csv"
echo ""
echo "Final report will be in:"
echo "  $RESULTS_ROOT/report/"
echo ""
echo "To check report status:"
echo "  tail -f jobs/outputs/build_report.log"
echo ""

