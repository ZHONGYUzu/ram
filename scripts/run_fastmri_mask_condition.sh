#!/bin/bash

# Shared body for ram-004 through ram-009. One invocation is one paper-table cell.

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ "$#" -ne 5 ]]; then
    echo "Usage: $0 RUN_ID MASK_MODE ACCELERATION CENTER_COLUMNS CENTER_FRACTION" >&2
    exit 2
fi

RUN_ID="$1"
MASK_MODE="$2"
ACCELERATION="$3"
CENTER_COLUMNS="$4"
CENTER_FRACTION="$5"
REPO_ROOT=/home/students/studxuzho1/ram-fastmri-brain-adapter
RUN_ROOT="$REPO_ROOT/runs/$RUN_ID"
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
DATA_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
CHECKPOINT_SHA256=1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
EXPECTED_BRANCH=feature/fastmri-brain-adapter
CONDITION="${MASK_MODE}-r${ACCELERATION}"
RUN_LOG="$RUN_ROOT/logs/run-${SLURM_JOB_ID}.log"
SUMMARY_DIR="$RUN_ROOT/summary/job-${SLURM_JOB_ID}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/output" "$RUN_ROOT/logs" "$RUN_ROOT/summary"
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "Expected branch $EXPECTED_BRANCH" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to run with uncommitted repository changes." >&2
    git status --short >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Cached checkpoint missing; downloads are disabled: $CHECKPOINT" >&2
    exit 2
fi
if [[ ! -f "$RUN_ROOT/cases.txt" ]]; then
    echo "Missing $RUN_ROOT/cases.txt" >&2
    exit 2
fi
if [[ "$(grep -cvE '^[[:space:]]*(#|$)' "$RUN_ROOT/cases.txt")" -ne 5 ]]; then
    echo "$RUN_ID requires exactly five cases" >&2
    exit 2
fi
if [[ -e "$SUMMARY_DIR" ]]; then
    echo "Refusing to overwrite $SUMMARY_DIR" >&2
    exit 2
fi

if [[ "$MASK_MODE" == "polynomial-vd-random" ]]; then
    SEEDS=(0 1 2)
else
    SEEDS=(0)
fi

cat > "$RUN_ROOT/config.yaml" <<EOF
experiment:
  id: $RUN_ID
  parent_run: ram-003
  condition: $CONDITION
  slurm_job_id: ${SLURM_JOB_ID}
  git_branch: $(git branch --show-current)
  git_commit: $(git rev-parse HEAD)
  downloads_allowed: false
data:
  dataset: fastMRI brain multicoil validation
  data_root: $DATA_ROOT
  cases_file: $RUN_ROOT/cases.txt
  case_count: 5
  slices_per_case: 3
  slice_selection: equally spaced within the 15%-85% non-edge range
preprocessing:
  coil_combination: volume-level ESC/VCC
  esc_max_iterations: 60
  normalization: zero-filled magnitude p99.5 per slice
mask:
  dimensionality: 1D Cartesian
  mode: $MASK_MODE
  nominal_acceleration: $ACCELERATION
  image_width: 320
  sampled_columns: $(( (320 + ACCELERATION / 2) / ACCELERATION ))
  center_columns: $CENTER_COLUMNS
  center_fraction: $CENTER_FRACTION
  polynomial_exponent: 4.0
  density_floor: 1.0e-6
  seeds: [$(IFS=,; echo "${SEEDS[*]}")]
  per_case_seed: sha256(filename + seed)
model:
  implementation: deepinv.models.RAM
  minimum_deepinv_version: 0.4.1
  checkpoint: $CHECKPOINT
  checkpoint_sha256: $CHECKPOINT_SHA256
  noise_sigma: 0.0005
output:
  root: $RUN_ROOT/output
  summary: $SUMMARY_DIR
  previews_per_replicate: 6
logs:
  run: $RUN_LOG
  stdout: $RUN_ROOT/logs/slurm-${SLURM_JOB_ID}.out
  stderr: $RUN_ROOT/logs/slurm-${SLURM_JOB_ID}.err
EOF

echo "Job ID: $SLURM_JOB_ID" | tee "$RUN_LOG"
echo "Git commit: $(git rev-parse HEAD)" | tee -a "$RUN_LOG"
echo "Run root: $RUN_ROOT" | tee -a "$RUN_LOG"
echo "Condition: $CONDITION" | tee -a "$RUN_LOG"
nvidia-smi | tee -a "$RUN_LOG"
"$PYTHON_BIN" -c 'import torch, deepinv, ram.adapters; print("Torch:", torch.__version__); print("DeepInverse:", deepinv.__version__); print("RAM:", ram.__file__); print("CUDA:", torch.version.cuda)' | tee -a "$RUN_LOG"

OUTPUT_DIRS=()
for seed in "${SEEDS[@]}"; do
    if [[ "$MASK_MODE" == "polynomial-vd-random" ]]; then
        replicate="seed${seed}"
    else
        replicate="deterministic"
    fi
    output_dir="$RUN_ROOT/output/${CONDITION}-${replicate}-job${SLURM_JOB_ID}"
    if [[ -e "$output_dir" ]]; then
        echo "Refusing to overwrite $output_dir" >&2
        exit 2
    fi
    echo "Starting replicate: $replicate" | tee -a "$RUN_LOG"
    "$PYTHON_BIN" scripts/benchmark_fastmri_brain_vcc_ram.py \
        --data-root "$DATA_ROOT" \
        --cases-file "$RUN_ROOT/cases.txt" \
        --output-dir "$output_dir" \
        --checkpoint "$CHECKPOINT" \
        --checkpoint-sha256 "$CHECKPOINT_SHA256" \
        --acceleration "$ACCELERATION" \
        --center-fraction "$CENTER_FRACTION" \
        --center-columns "$CENTER_COLUMNS" \
        --mask-mode "$MASK_MODE" \
        --vd-exponent 4.0 \
        --vd-floor 1e-6 \
        --normalization-mode zf-percentile \
        --normalization-percentile 99.5 \
        --noise-sigma 0.0005 \
        --max-volumes 5 \
        --slices-per-volume 3 \
        --edge-fraction 0.15 \
        --esc-max-iterations 60 \
        --seed "$seed" \
        --save-previews 6 \
        --device cuda 2>&1 | tee -a "$RUN_LOG"
    cp "$RUN_LOG" "$output_dir/run.log"
    OUTPUT_DIRS+=(--input-dir "$output_dir")
done

"$PYTHON_BIN" scripts/summarize_fastmri_mask_run.py \
    --condition "$CONDITION" \
    --output-dir "$SUMMARY_DIR" \
    "${OUTPUT_DIRS[@]}" 2>&1 | tee -a "$RUN_LOG"
cp "$RUN_LOG" "$SUMMARY_DIR/run.log"
echo "Finished: $(date --iso-8601=seconds)" | tee -a "$RUN_LOG"
