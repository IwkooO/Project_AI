#!/usr/bin/env bash
set -euo pipefail

# Re-run Stage-1 (Phase-1) sweeps from existing output logs, but with:
#  - per-run checkpoint directories (no collisions)
#  - per-run SLURM log directories (no collisions)
#  - explicit WANDB run names (no collisions)
#
# It scans:
#   jobs/outputs/phase1/*.out
#   jobs/outputs/phase1_simple/*.out
#   jobs/outputs/phase1_textinit/*.out
#
# Usage:
#   bash jobs/submit_stage1_rerun_from_outputs.sh
#
# Optional:
#   DRY_RUN=1 bash jobs/submit_stage1_rerun_from_outputs.sh
#   RUN_ROOT=20251225_rerun bash jobs/submit_stage1_rerun_from_outputs.sh

DRY_RUN="${DRY_RUN:-0}"
RUN_ROOT="${RUN_ROOT:-rerun_$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-auto}"  # auto|manual
GROUPS="${GROUPS:-phase1,phase1_simple,phase1_textinit}"  # comma-separated
VERIFY="${VERIFY:-1}"            # 1 = verify configs against original logs before submitting
VERIFY_STRICT="${VERIFY_STRICT:-1}"  # 1 = fail hard on verification mismatch
ALLOW_MISSING_LOGS="${ALLOW_MISSING_LOGS:-0}" # 0 = fail if we can't find a matching log for a manual config

LOG_ROOT="jobs/outputs/reruns_stage1/${RUN_ROOT}"
CKPT_ROOT="checkpoints/reruns_stage1/${RUN_ROOT}"

PHASE1_SIMPLE_JOB="jobs/train_cbm_stage_1.job"
PHASE1_TEXTINIT_JOB="jobs/train_cbm_stage_1_textinit_interpretable.job"

# "phase1/" runs are also based on the simple job script (256-dim), just a different naming convention.
PHASE1_JOB="$PHASE1_SIMPLE_JOB"

# Common knobs (keep consistent with existing sweeps)
ATTN_DIV_TOPM_DEFAULT=8
ATTN_DIV_WARMUP_DEFAULT=5
ATTN_DIV_MODE_DEFAULT="offdiag"
STK_K_MASK_DEFAULT=1
MIX_LOCAL_KERNEL_DEFAULT=5
TAU_DEFAULT=0.25
TOPK_DEFAULT=6
PROJ_TYPE_DEFAULT="simple"

mkdir -p "$LOG_ROOT" "$CKPT_ROOT"
mkdir -p "$LOG_ROOT/phase1" "$LOG_ROOT/phase1_simple" "$LOG_ROOT/phase1_textinit"
mkdir -p "$CKPT_ROOT/phase1" "$CKPT_ROOT/phase1_simple" "$CKPT_ROOT/phase1_textinit"

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

has_group() {
  # Usage: has_group "phase1_simple"
  local g="$1"
  [[ ",${GROUPS}," == *",${g},"* ]]
}

fail_or_warn() {
  local msg="$1"
  if [[ "${VERIFY_STRICT}" == "1" ]]; then
    echo "ERROR: ${msg}" >&2
    exit 1
  else
    echo "WARNING: ${msg}" >&2
  fi
}

verify_phase1_simple_log() {
  # For phase1_simple logs we can strongly verify the full tag because wandb prints it:
  #   wandb: Syncing run phase1_tau0.25_k6_stk0.40_lk5_div1.2
  local log_file="$1"
  local expected_tag="$2" # tauX_kY_stkZ_lkW_divV (no "phase1_" prefix)

  if [[ ! -f "$log_file" ]]; then
    if [[ "${ALLOW_MISSING_LOGS}" == "1" ]]; then
      echo "WARNING: missing log file for verification: ${log_file}" >&2
      return 0
    fi
    fail_or_warn "missing log file for verification: ${log_file}"
    return 1
  fi

  local expected="wandb: Syncing run phase1_${expected_tag}"
  if ! grep -qF "$expected" "$log_file"; then
    fail_or_warn "phase1_simple verification failed for ${log_file}: expected to find '${expected}'"
    return 1
  fi
  return 0
}

verify_textinit_log() {
  # Text-init logs do NOT include the diversity *weight* as a string.
  # They do include a RUN_TAG line with tau/topk/stk/lk:
  #   RUN_TAG=tau0.18_k4_stk0.45_lk5
  # We verify that part exactly. (div is verified via filename-based tag only.)
  local log_file="$1"
  local expected_runtag="$2" # tauX_kY_stkZ_lkW (no div)

  if [[ ! -f "$log_file" ]]; then
    if [[ "${ALLOW_MISSING_LOGS}" == "1" ]]; then
      echo "WARNING: missing log file for verification: ${log_file}" >&2
      return 0
    fi
    fail_or_warn "missing log file for verification: ${log_file}"
    return 1
  fi

  local expected="RUN_TAG=${expected_runtag}"
  if ! grep -qF "$expected" "$log_file"; then
    fail_or_warn "phase1_textinit verification failed for ${log_file}: expected to find '${expected}'"
    return 1
  fi
  return 0
}

find_latest_log_or_fail() {
  # Usage: find_latest_log_or_fail "jobs/outputs/phase1_simple/phase1_tau0.25_k6_*.out"
  local pattern="$1"
  local found
  found="$(ls -t $pattern 2>/dev/null | head -n 1 || true)"
  if [[ -z "$found" ]]; then
    if [[ "${ALLOW_MISSING_LOGS}" == "1" ]]; then
      echo ""
      return 0
    fi
    fail_or_warn "no matching log found for pattern: ${pattern}"
    return 1
  fi
  echo "$found"
}

# ---------------------------------------------------------------------------
# MANUAL CONFIG LISTS (explicit, reviewed)
#
# Use MODE=manual to submit these exact configurations (recommended when you
# want to double-check correctness and avoid relying on filename parsing).
#
# These correspond to the configs we observed in your logs:
#  - phase1/:         phase1_divX_stkY_*.out
#  - phase1_simple/:  phase1_tauX_kY_stkZ_lkW_divV_*.out
#  - phase1_textinit/:textinit_tauX_kY_stkZ_lkW_divV_*.out
# ---------------------------------------------------------------------------

submit_manual_phase1() {
  # Format: "div stk"
  local configs=(
    "0.0 0.0"
    "0.0 0.40"
    "0.005 0.0"
    "0.01 0.30"
    "0.01 0.50"
    "0.02 0.30"
    "0.05 0.0"
    "0.05 0.50"
    "0.2 0.40"
    "0.5 0.35"
  )

  for c in "${configs[@]}"; do
    local div stk
    div="$(awk '{print $1}' <<<"$c")"
    stk="$(awk '{print $2}' <<<"$c")"

    local tag="div${div}_stk${stk}"
    local out_dir="${CKPT_ROOT}/phase1/${tag}"
    local out_log="${LOG_ROOT}/phase1/phase1_${tag}_%j.out"
    local wandb="rerun_phase1_${tag}_${RUN_ROOT}"

    local export_vars
    export_vars="ALL,TAU=${TAU_DEFAULT},TOPK=${TOPK_DEFAULT},STK_MASK_PROB=${stk},STK_K_MASK=${STK_K_MASK_DEFAULT},MIX_LOCAL_KERNEL=${MIX_LOCAL_KERNEL_DEFAULT},PROJ_TYPE=${PROJ_TYPE_DEFAULT},ATTN_DIV_W=${div},ATTN_DIV_TOPM=${ATTN_DIV_TOPM_DEFAULT},ATTN_DIV_WARMUP=${ATTN_DIV_WARMUP_DEFAULT},ATTN_DIV_MODE=${ATTN_DIV_MODE_DEFAULT},WANDB_RUN_NAME=${wandb},OUTPUT_DIR=${out_dir}"

    submit "sbatch --output=${out_log} --export=${export_vars} ${PHASE1_JOB}"
  done
}

submit_manual_phase1_simple() {
  # Format: "tau topk stk lk div"
  local configs=(
    "0.20 6 0.40 5 0.5"
    "0.25 6 0.40 5 0.3"
    "0.25 6 0.40 5 0.5"
    "0.25 6 0.50 5 0.5"
    "0.25 6 0.40 5 0.6"
    "0.25 7 0.40 5 0.5"
    "0.25 8 0.40 5 0.5"
    "0.30 6 0.40 5 0.5"
    "0.25 6 0.40 5 0.8"
    "0.25 6 0.40 5 1.0"
    "0.25 6 0.40 5 1.2"
  )

  for c in "${configs[@]}"; do
    local tau k stk lk div
    tau="$(awk '{print $1}' <<<"$c")"
    k="$(awk '{print $2}' <<<"$c")"
    stk="$(awk '{print $3}' <<<"$c")"
    lk="$(awk '{print $4}' <<<"$c")"
    div="$(awk '{print $5}' <<<"$c")"

    local tag="tau${tau}_k${k}_stk${stk}_lk${lk}_div${div}"

    if [[ "${VERIFY}" == "1" ]]; then
      local src_log
      src_log="$(find_latest_log_or_fail "jobs/outputs/phase1_simple/phase1_${tag}_*.out")"
      if [[ -n "${src_log}" ]]; then
        verify_phase1_simple_log "${src_log}" "${tag}"
      fi
    fi

    local out_dir="${CKPT_ROOT}/phase1_simple/${tag}"
    local out_log="${LOG_ROOT}/phase1_simple/phase1_${tag}_%j.out"
    local wandb="rerun_phase1_simple_${tag}_${RUN_ROOT}"

    local export_vars
    export_vars="ALL,TAU=${tau},TOPK=${k},STK_MASK_PROB=${stk},STK_K_MASK=${STK_K_MASK_DEFAULT},MIX_LOCAL_KERNEL=${lk},PROJ_TYPE=${PROJ_TYPE_DEFAULT},ATTN_DIV_W=${div},ATTN_DIV_TOPM=${ATTN_DIV_TOPM_DEFAULT},ATTN_DIV_WARMUP=${ATTN_DIV_WARMUP_DEFAULT},ATTN_DIV_MODE=${ATTN_DIV_MODE_DEFAULT},WANDB_RUN_NAME=${wandb},OUTPUT_DIR=${out_dir}"

    submit "sbatch --output=${out_log} --export=${export_vars} ${PHASE1_SIMPLE_JOB}"
  done
}

submit_manual_phase1_textinit() {
  # Format: "tau topk stk lk div"
  local configs=(
    "0.18 3 0.45 5 0.15"
    "0.20 5 0.40 5 0.15"
    "0.12 3 0.50 5 0.25"
    "0.18 4 0.45 5 0.15"
    "0.18 4 0.45 5 0.25"
    "0.18 4 0.50 5 0.15"
    "0.15 3 0.50 5 0.20"
  )

  for c in "${configs[@]}"; do
    local tau k stk lk div
    tau="$(awk '{print $1}' <<<"$c")"
    k="$(awk '{print $2}' <<<"$c")"
    stk="$(awk '{print $3}' <<<"$c")"
    lk="$(awk '{print $4}' <<<"$c")"
    div="$(awk '{print $5}' <<<"$c")"

    local tag="tau${tau}_k${k}_stk${stk}_lk${lk}_div${div}"
    local runtag="tau${tau}_k${k}_stk${stk}_lk${lk}"

    if [[ "${VERIFY}" == "1" ]]; then
      local src_log
      src_log="$(find_latest_log_or_fail "jobs/outputs/phase1_textinit/textinit_${tag}_*.out")"
      if [[ -n "${src_log}" ]]; then
        verify_textinit_log "${src_log}" "${runtag}"
      fi
    fi

    local out_dir="${CKPT_ROOT}/phase1_textinit/${tag}"
    local out_log="${LOG_ROOT}/phase1_textinit/textinit_${tag}_%j.out"
    local wandb="rerun_phase1_textinit_${tag}_${RUN_ROOT}"

    local export_vars
    export_vars="ALL,TAU=${tau},TOPK=${k},STK_MASK_PROB=${stk},STK_K_MASK=${STK_K_MASK_DEFAULT},MIX_LOCAL_KERNEL=${lk},ATTN_DIV_W=${div},ATTN_DIV_TOPM=${ATTN_DIV_TOPM_DEFAULT},ATTN_DIV_WARMUP=${ATTN_DIV_WARMUP_DEFAULT},ATTN_DIV_MODE=${ATTN_DIV_MODE_DEFAULT},WANDB_RUN_NAME=${wandb},OUTPUT_DIR=${out_dir}"

    submit "sbatch --output=${out_log} --export=${export_vars} ${PHASE1_TEXTINIT_JOB}"
  done
}

submit_manual_all() {
  echo ""
  if has_group "phase1"; then
    echo "Submitting (manual) phase1/ ..."
    submit_manual_phase1
    echo ""
  fi
  if has_group "phase1_simple"; then
    echo "Submitting (manual) phase1_simple/ ..."
    submit_manual_phase1_simple
    echo ""
  fi
  if has_group "phase1_textinit"; then
    echo "Submitting (manual) phase1_textinit/ ..."
    submit_manual_phase1_textinit
    echo ""
  fi
}

dedup_and_submit_phase1_simple() {
  shopt -s nullglob
  declare -A seen=()
  local files=(jobs/outputs/phase1_simple/*.out)
  shopt -u nullglob

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No files found in jobs/outputs/phase1_simple/; skipping."
    return 0
  fi

  for f in "${files[@]}"; do
    local base
    base="$(basename "$f")"

    # Example: phase1_tau0.25_k6_stk0.40_lk5_div1.2_17868889.out
    if [[ "$base" =~ ^phase1_tau([0-9.]+)_k([0-9]+)_stk([0-9.]+)_lk([0-9]+)_div([0-9.]+)_ ]]; then
      local tau="${BASH_REMATCH[1]}"
      local k="${BASH_REMATCH[2]}"
      local stk="${BASH_REMATCH[3]}"
      local lk="${BASH_REMATCH[4]}"
      local div="${BASH_REMATCH[5]}"

      local tag="tau${tau}_k${k}_stk${stk}_lk${lk}_div${div}"
      if [[ -n "${seen[$tag]:-}" ]]; then
        continue
      fi
      seen[$tag]=1

      if [[ "${VERIFY}" == "1" ]]; then
        verify_phase1_simple_log "$f" "$tag"
      fi

      local out_dir="${CKPT_ROOT}/phase1_simple/${tag}"
      local out_log="${LOG_ROOT}/phase1_simple/phase1_${tag}_%j.out"
      local wandb="rerun_phase1_simple_${tag}_${RUN_ROOT}"

      local export_vars
      export_vars="ALL,TAU=${tau},TOPK=${k},STK_MASK_PROB=${stk},STK_K_MASK=${STK_K_MASK_DEFAULT},MIX_LOCAL_KERNEL=${lk},PROJ_TYPE=${PROJ_TYPE_DEFAULT},ATTN_DIV_W=${div},ATTN_DIV_TOPM=${ATTN_DIV_TOPM_DEFAULT},ATTN_DIV_WARMUP=${ATTN_DIV_WARMUP_DEFAULT},ATTN_DIV_MODE=${ATTN_DIV_MODE_DEFAULT},WANDB_RUN_NAME=${wandb},OUTPUT_DIR=${out_dir}"

      submit "sbatch --output=${out_log} --export=${export_vars} ${PHASE1_SIMPLE_JOB}"
    fi
  done
}

dedup_and_submit_phase1() {
  shopt -s nullglob
  declare -A seen=()
  local files=(jobs/outputs/phase1/*.out)
  shopt -u nullglob

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No files found in jobs/outputs/phase1/; skipping."
    return 0
  fi

  for f in "${files[@]}"; do
    local base
    base="$(basename "$f")"

    # Example: phase1_div0.2_stk0.40_17865203.out
    if [[ "$base" =~ ^phase1_div([0-9.]+)_stk([0-9.]+)_ ]]; then
      local div="${BASH_REMATCH[1]}"
      local stk="${BASH_REMATCH[2]}"

      local tag="div${div}_stk${stk}"
      if [[ -n "${seen[$tag]:-}" ]]; then
        continue
      fi
      seen[$tag]=1

      local out_dir="${CKPT_ROOT}/phase1/${tag}"
      local out_log="${LOG_ROOT}/phase1/phase1_${tag}_%j.out"
      local wandb="rerun_phase1_${tag}_${RUN_ROOT}"

      local export_vars
      export_vars="ALL,TAU=${TAU_DEFAULT},TOPK=${TOPK_DEFAULT},STK_MASK_PROB=${stk},STK_K_MASK=${STK_K_MASK_DEFAULT},MIX_LOCAL_KERNEL=${MIX_LOCAL_KERNEL_DEFAULT},PROJ_TYPE=${PROJ_TYPE_DEFAULT},ATTN_DIV_W=${div},ATTN_DIV_TOPM=${ATTN_DIV_TOPM_DEFAULT},ATTN_DIV_WARMUP=${ATTN_DIV_WARMUP_DEFAULT},ATTN_DIV_MODE=${ATTN_DIV_MODE_DEFAULT},WANDB_RUN_NAME=${wandb},OUTPUT_DIR=${out_dir}"

      submit "sbatch --output=${out_log} --export=${export_vars} ${PHASE1_JOB}"
    fi
  done
}

dedup_and_submit_phase1_textinit() {
  shopt -s nullglob
  declare -A seen=()
  local files=(jobs/outputs/phase1_textinit/*.out)
  shopt -u nullglob

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No files found in jobs/outputs/phase1_textinit/; skipping."
    return 0
  fi

  for f in "${files[@]}"; do
    local base
    base="$(basename "$f")"

    # Example: textinit_tau0.18_k4_stk0.45_lk5_div0.15_17867603.out
    if [[ "$base" =~ ^textinit_tau([0-9.]+)_k([0-9]+)_stk([0-9.]+)_lk([0-9]+)_div([0-9.]+)_ ]]; then
      local tau="${BASH_REMATCH[1]}"
      local k="${BASH_REMATCH[2]}"
      local stk="${BASH_REMATCH[3]}"
      local lk="${BASH_REMATCH[4]}"
      local div="${BASH_REMATCH[5]}"

      local tag="tau${tau}_k${k}_stk${stk}_lk${lk}_div${div}"
      local runtag="tau${tau}_k${k}_stk${stk}_lk${lk}"
      if [[ -n "${seen[$tag]:-}" ]]; then
        continue
      fi
      seen[$tag]=1

      if [[ "${VERIFY}" == "1" ]]; then
        verify_textinit_log "$f" "$runtag"
      fi

      local out_dir="${CKPT_ROOT}/phase1_textinit/${tag}"
      local out_log="${LOG_ROOT}/phase1_textinit/textinit_${tag}_%j.out"
      local wandb="rerun_phase1_textinit_${tag}_${RUN_ROOT}"

      local export_vars
      export_vars="ALL,TAU=${tau},TOPK=${k},STK_MASK_PROB=${stk},STK_K_MASK=${STK_K_MASK_DEFAULT},MIX_LOCAL_KERNEL=${lk},ATTN_DIV_W=${div},ATTN_DIV_TOPM=${ATTN_DIV_TOPM_DEFAULT},ATTN_DIV_WARMUP=${ATTN_DIV_WARMUP_DEFAULT},ATTN_DIV_MODE=${ATTN_DIV_MODE_DEFAULT},WANDB_RUN_NAME=${wandb},OUTPUT_DIR=${out_dir}"

      submit "sbatch --output=${out_log} --export=${export_vars} ${PHASE1_TEXTINIT_JOB}"
    fi
  done
}

echo "============================================================"
echo "Stage-1 rerun submitter"
echo "RUN_ROOT:   ${RUN_ROOT}"
echo "LOG_ROOT:   ${LOG_ROOT}"
echo "CKPT_ROOT:  ${CKPT_ROOT}"
echo "DRY_RUN:    ${DRY_RUN}"
echo "MODE:       ${MODE}"
echo "GROUPS:     ${GROUPS}"
echo "VERIFY:     ${VERIFY} (STRICT=${VERIFY_STRICT}, ALLOW_MISSING_LOGS=${ALLOW_MISSING_LOGS})"
echo "============================================================"

if [[ "$MODE" == "manual" ]]; then
  submit_manual_all
else
  if has_group "phase1"; then
    echo ""
    echo "Submitting phase1/ (div+stk) from existing logs..."
    dedup_and_submit_phase1
  fi

  if has_group "phase1_simple"; then
    echo ""
    echo "Submitting phase1_simple/ (tau+k+stk+lk+div) from existing logs..."
    dedup_and_submit_phase1_simple
  fi

  if has_group "phase1_textinit"; then
    echo ""
    echo "Submitting phase1_textinit/ (tau+k+stk+lk+div) from existing logs..."
    dedup_and_submit_phase1_textinit
  fi
fi

echo ""
echo "============================================================"
echo "Done submitting reruns."
echo "Logs:       ${LOG_ROOT}"
echo "Checkpoints:${CKPT_ROOT}"
echo "Check status: squeue -u \$USER"
echo "============================================================"


