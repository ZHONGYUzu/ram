#!/bin/bash

# ram-010: one fastMRI brain volume, fastMRI random Cartesian R8/R16/R24, ZF p99.5.

#SBATCH --job-name=ram010_rand
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-010/logs/slurm-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-010/logs/slurm-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_ROOT=/home/students/studxuzho1/ram-fastmri-brain-adapter
RUN_ROOT="$REPO_ROOT/runs/ram-010"
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
DATA_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
CHECKPOINT_SHA256=1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
EXPECTED_BRANCH=feature/fastmri-brain-adapter
CASE=file_brain_AXFLAIR_200_6002462.h5
RUN_LOG="$RUN_ROOT/logs/run-${SLURM_JOB_ID}.log"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/output" "$RUN_ROOT/logs"
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
if [[ ! -f "$DATA_ROOT/$CASE" ]]; then
    echo "Missing fastMRI volume: $DATA_ROOT/$CASE" >&2
    exit 2
fi

printf '%s\n' "$CASE" > "$RUN_ROOT/cases.txt"

cat > "$RUN_ROOT/config.yaml" <<EOF
experiment:
  id: ram-010
  parent_run: ram-002
  purpose: single-volume fastMRI random Cartesian acceleration comparison with ZF p99.5 scaling
  excludes_prior_job: 42509 (wrong exact-equispaced mask)
  slurm_job_id: ${SLURM_JOB_ID}
  git_branch: $(git branch --show-current)
  git_commit: $(git rev-parse HEAD)
  downloads_allowed: false
data:
  dataset: fastMRI brain multicoil validation
  data_root: $DATA_ROOT
  case: $CASE
  slices_per_volume: 3
  selected_slices: [2, 8, 13]
  slice_selection: equally spaced within the 15%-85% non-edge range
preprocessing:
  coil_combination: volume-level ESC/VCC
  esc_max_iterations: 60
  normalization: zero-filled magnitude p99.5 per slice
mask:
  dimensionality: 1D Cartesian
  mode: fastmri-random
  nominal_accelerations: [8, 16, 24]
  image_width: 320
  center_fractions: [0.04, 0.02, 0.01]
  base_seed: 0
  per_volume_seed: sha256(filename + base_seed)
model:
  implementation: deepinv.models.RAM
  minimum_deepinv_version: 0.4.1
  checkpoint: $CHECKPOINT
  checkpoint_sha256: $CHECKPOINT_SHA256
  noise_sigma: 0.0005
  post_ram_data_consistency: false
output:
  root: $RUN_ROOT/output
  previews_per_acceleration: 3
logs:
  run: $RUN_LOG
  stdout: $RUN_ROOT/logs/slurm-${SLURM_JOB_ID}.out
  stderr: $RUN_ROOT/logs/slurm-${SLURM_JOB_ID}.err
EOF

echo "Job ID: $SLURM_JOB_ID" | tee "$RUN_LOG"
echo "Git commit: $(git rev-parse HEAD)" | tee -a "$RUN_LOG"
echo "Run root: $RUN_ROOT" | tee -a "$RUN_LOG"
echo "Case: $CASE" | tee -a "$RUN_LOG"
nvidia-smi | tee -a "$RUN_LOG"
"$PYTHON_BIN" -c 'import torch, deepinv, ram.adapters; print("Torch:", torch.__version__); print("DeepInverse:", deepinv.__version__); print("RAM:", ram.__file__); print("CUDA:", torch.version.cuda)' | tee -a "$RUN_LOG"

ACCELERATIONS=(8 16 24)
CENTER_FRACTIONS=(0.04 0.02 0.01)

for index in "${!ACCELERATIONS[@]}"; do
    acceleration="${ACCELERATIONS[$index]}"
    center_fraction="${CENTER_FRACTIONS[$index]}"
    output_dir="$RUN_ROOT/output/fastmri-random-r${acceleration}-seed0-job${SLURM_JOB_ID}"
    if [[ -e "$output_dir" ]]; then
        echo "Refusing to overwrite $output_dir" >&2
        exit 2
    fi

    echo "Starting R${acceleration}: fastMRI random Cartesian, center_fraction=${center_fraction}, base_seed=0" | tee -a "$RUN_LOG"
    "$PYTHON_BIN" scripts/benchmark_fastmri_brain_vcc_ram.py \
        --data-root "$DATA_ROOT" \
        --cases-file "$RUN_ROOT/cases.txt" \
        --output-dir "$output_dir" \
        --checkpoint "$CHECKPOINT" \
        --checkpoint-sha256 "$CHECKPOINT_SHA256" \
        --acceleration "$acceleration" \
        --center-fraction "$center_fraction" \
        --mask-mode fastmri-random \
        --normalization-mode zf-percentile \
        --normalization-percentile 99.5 \
        --noise-sigma 0.0005 \
        --max-volumes 1 \
        --slices-per-volume 3 \
        --edge-fraction 0.15 \
        --esc-max-iterations 60 \
        --seed 0 \
        --save-previews 3 \
        --device cuda 2>&1 | tee -a "$RUN_LOG"
    cp "$RUN_LOG" "$output_dir/run.log"
done

echo "Finished: $(date --iso-8601=seconds)" | tee -a "$RUN_LOG"
