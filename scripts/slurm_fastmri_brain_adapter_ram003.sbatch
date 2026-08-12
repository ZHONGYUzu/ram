#!/bin/bash

# ram-003: five-case generalization check using the validated ram-002 settings.

#SBATCH --job-name=ram003_fmri
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-003/logs/slurm-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-003/logs/slurm-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_ROOT=/home/students/studxuzho1/ram-fastmri-brain-adapter
RUN_ROOT="$REPO_ROOT/runs/ram-003"
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
DATA_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
EXPECTED_BRANCH=feature/fastmri-brain-adapter
CHECKPOINT_SHA256=1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
RUN_ID="five-cases-r4-job${SLURM_JOB_ID}"
OUTPUT_DIR="$RUN_ROOT/output/$RUN_ID"
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
if [[ ! -f "$RUN_ROOT/cases.txt" ]]; then
    echo "Missing $RUN_ROOT/cases.txt" >&2
    exit 2
fi
if [[ "$(grep -cvE '^[[:space:]]*(#|$)' "$RUN_ROOT/cases.txt")" -ne 5 ]]; then
    echo "ram-003 requires exactly five cases" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite $OUTPUT_DIR" >&2
    exit 2
fi

cat > "$RUN_ROOT/config.yaml" <<EOF
experiment:
  id: ram-003
  parent_run: ram-002
  purpose: five-case cross-acquisition generalization check
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
  slice_selection: equally spaced within the 15%-85% non-edge range
preprocessing:
  coil_combination: volume-level ESC/VCC
  esc_max_iterations: 60
  normalization: zero-filled magnitude p99.5 per slice
mask:
  type: exact-random Cartesian
  acceleration: 4
  image_width: 320
  sampled_columns: 80
  center_fraction: 0.08
  low_frequency_columns: 26
  seed: 0
  per_case_seed: sha256(filename + seed)
model:
  implementation: deepinv.models.RAM
  minimum_deepinv_version: 0.4.1
  checkpoint: $CHECKPOINT
  checkpoint_sha256: $CHECKPOINT_SHA256
  noise_sigma: 0.0005
output:
  directory: $OUTPUT_DIR
  previews: 6
  slice_metrics: true
  summary: true
logs:
  run: $RUN_LOG
  stdout: $RUN_ROOT/logs/slurm-${SLURM_JOB_ID}.out
  stderr: $RUN_ROOT/logs/slurm-${SLURM_JOB_ID}.err
EOF

echo "Job ID: $SLURM_JOB_ID"
echo "Git commit: $(git rev-parse HEAD)"
echo "Run root: $RUN_ROOT"
echo "Output: $OUTPUT_DIR"
nvidia-smi
"$PYTHON_BIN" -c 'import torch, deepinv, ram.adapters; print("Torch:", torch.__version__); print("DeepInverse:", deepinv.__version__); print("RAM:", ram.__file__); print("CUDA:", torch.version.cuda)'

"$PYTHON_BIN" scripts/benchmark_fastmri_brain_vcc_ram.py \
    --data-root "$DATA_ROOT" \
    --cases-file "$RUN_ROOT/cases.txt" \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --checkpoint-sha256 "$CHECKPOINT_SHA256" \
    --acceleration 4 \
    --center-fraction 0.08 \
    --mask-mode exact-random \
    --normalization-mode zf-percentile \
    --normalization-percentile 99.5 \
    --noise-sigma 0.0005 \
    --max-volumes 5 \
    --slices-per-volume 3 \
    --edge-fraction 0.15 \
    --esc-max-iterations 60 \
    --seed 0 \
    --save-previews 6 \
    --device cuda 2>&1 | tee "$RUN_LOG"

cp "$RUN_LOG" "$OUTPUT_DIR/run.log"
echo "Finished: $(date --iso-8601=seconds)"
