#!/bin/bash

# Shared body for the one-acceleration-per-run RAM paper-distribution benchmarks.

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 RUN_ID ACCELERATION CENTER_FRACTION" >&2
    exit 2
fi

RUN_ID="$1"
ACCELERATION="$2"
CENTER_FRACTION="$3"
REPO_ROOT=/home/students/studxuzho1/ram-fastmri-brain-adapter
RUN_ROOT="$REPO_ROOT/runs/$RUN_ID"
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
DATA_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
CHECKPOINT_SHA256=1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
EXPECTED_BRANCH=feature/fastmri-brain-adapter
RUN_LOG="$RUN_ROOT/logs/run-${SLURM_JOB_ID}.log"
SUMMARY_DIR="$RUN_ROOT/summary/job-${SLURM_JOB_ID}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

case "$RUN_ID:$ACCELERATION:$CENTER_FRACTION" in
    ram-013:4:0.08|ram-014:8:0.04)
        PURPOSE="approximate the RAM paper fastMRI MRI training distribution at R${ACCELERATION}"
        PAPER_RELATION="in-distribution acceleration"
        ;;
    ram-015:16:0.02|ram-016:24:0.01)
        PURPOSE="test high-acceleration generalization beyond the RAM paper training distribution at R${ACCELERATION}"
        PAPER_RELATION="out-of-distribution acceleration"
        ;;
    *) echo "Unsupported paper condition: $RUN_ID R$ACCELERATION CF=$CENTER_FRACTION" >&2; exit 2 ;;
esac

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
if [[ -e "$SUMMARY_DIR" ]]; then
    echo "Refusing to overwrite $SUMMARY_DIR" >&2
    exit 2
fi

cat > "$RUN_ROOT/cases.txt" <<EOF
file_brain_AXFLAIR_200_6002462.h5
file_brain_AXFLAIR_200_6002471.h5
file_brain_AXT1_201_6002725.h5
file_brain_AXT1POST_200_6001976.h5
file_brain_AXT1PRE_200_6002079.h5
EOF

while IFS= read -r case_name; do
    if [[ ! -f "$DATA_ROOT/$case_name" ]]; then
        echo "Missing fastMRI volume: $DATA_ROOT/$case_name" >&2
        exit 2
    fi
done < "$RUN_ROOT/cases.txt"

cat > "$RUN_ROOT/config.yaml" <<EOF
experiment:
  id: $RUN_ID
  parent_run: ram-003
  purpose: $PURPOSE
  paper_relation: $PAPER_RELATION
  normalization_note: zero-filled magnitude p99.5 normalization per slice
  slurm_job_id: ${SLURM_JOB_ID}
  git_branch: $(git branch --show-current)
  git_commit: $(git rev-parse HEAD)
  downloads_allowed: false
data:
  dataset: fastMRI brain multicoil validation
  data_root: $DATA_ROOT
  cases_file: $RUN_ROOT/cases.txt
  case_count: 5
  acquisitions: [AXFLAIR, AXFLAIR, AXT1, AXT1POST, AXT1PRE]
  slices_per_case: 3
  independent_slices: 15
  slice_selection: equally spaced within the 15%-85% non-edge range
preprocessing:
  coil_combination: volume-level ESC/VCC
  esc_max_iterations: 60
  image_shape: [320, 320]
  ram_tensor_shape_per_inference: [1, 2, 320, 320]
  complex_channels: [real, imaginary]
  normalization: zero-filled magnitude p99.5 per slice
mask:
  dimensionality: 1D Cartesian
  mode: fastmri-random
  nominal_acceleration: $ACCELERATION
  center_fraction: $CENTER_FRACTION
  base_seeds: [0, 1, 2]
  per_volume_seed: sha256(filename + base_seed)
model:
  implementation: deepinv.models.RAM
  minimum_deepinv_version: 0.4.1
  checkpoint: $CHECKPOINT
  checkpoint_sha256: $CHECKPOINT_SHA256
  model_call: model(y, physics)
  noise_model: Gaussian
  noise_sigma: 0.0005
  synthetic_noise_added: true
  post_ram_data_consistency: false
evaluation:
  inference_count: 45
  formula: 5 cases x 3 slices x 3 mask seeds
  metrics: [PSNR, SSIM, NMSE]
  metric_image: magnitude
output:
  root: $RUN_ROOT/output
  summary: $SUMMARY_DIR
  previews_per_seed: 3
logs:
  run: $RUN_LOG
  stdout: $RUN_ROOT/logs/slurm-${SLURM_JOB_ID}.out
  stderr: $RUN_ROOT/logs/slurm-${SLURM_JOB_ID}.err
EOF

echo "Job ID: $SLURM_JOB_ID" | tee "$RUN_LOG"
echo "Git commit: $(git rev-parse HEAD)" | tee -a "$RUN_LOG"
echo "Run root: $RUN_ROOT" | tee -a "$RUN_LOG"
echo "Condition: fastmri-random R${ACCELERATION}, center fraction ${CENTER_FRACTION}" | tee -a "$RUN_LOG"
nvidia-smi | tee -a "$RUN_LOG"
"$PYTHON_BIN" -c 'import torch, deepinv, ram.adapters; print("Torch:", torch.__version__); print("DeepInverse:", deepinv.__version__); print("RAM:", ram.__file__); print("CUDA:", torch.version.cuda)' | tee -a "$RUN_LOG"

SEEDS=(0 1 2)
OUTPUT_DIRS=()
for seed in "${SEEDS[@]}"; do
    output_dir="$RUN_ROOT/output/fastmri-random-r${ACCELERATION}-seed${seed}-job${SLURM_JOB_ID}"
    if [[ -e "$output_dir" ]]; then
        echo "Refusing to overwrite $output_dir" >&2
        exit 2
    fi
    echo "Starting R${ACCELERATION}, seed ${seed}" | tee -a "$RUN_LOG"
    "$PYTHON_BIN" scripts/benchmark_fastmri_brain_vcc_ram.py \
        --data-root "$DATA_ROOT" \
        --cases-file "$RUN_ROOT/cases.txt" \
        --output-dir "$output_dir" \
        --checkpoint "$CHECKPOINT" \
        --checkpoint-sha256 "$CHECKPOINT_SHA256" \
        --acceleration "$ACCELERATION" \
        --center-fraction "$CENTER_FRACTION" \
        --mask-mode fastmri-random \
        --normalization-mode zf-percentile \
        --normalization-percentile 99.5 \
        --noise-sigma 0.0005 \
        --max-volumes 5 \
        --slices-per-volume 3 \
        --edge-fraction 0.15 \
        --esc-max-iterations 60 \
        --seed "$seed" \
        --save-previews 3 \
        --device cuda 2>&1 | tee -a "$RUN_LOG"
    cp "$RUN_LOG" "$output_dir/run.log"
    OUTPUT_DIRS+=(--input-dir "$output_dir")
done

"$PYTHON_BIN" scripts/summarize_fastmri_mask_run.py \
    --condition "fastmri-random-r${ACCELERATION}" \
    --output-dir "$SUMMARY_DIR" \
    "${OUTPUT_DIRS[@]}" 2>&1 | tee -a "$RUN_LOG"
cp "$RUN_LOG" "$SUMMARY_DIR/run.log"
echo "Finished: $(date --iso-8601=seconds)" | tee -a "$RUN_LOG"
