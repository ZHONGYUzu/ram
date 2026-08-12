#!/bin/bash

# Run with: ACCELERATION=4 sbatch scripts/slurm_fastmri_brain_adapter.sbatch

#SBATCH --job-name=ram_fmri_adapter
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-001/logs/slurm-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-001/logs/slurm-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_ROOT=/home/students/studxuzho1/ram-fastmri-brain-adapter
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
DATA_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
RESULTS_ROOT=/home/students/studxuzho1/ram-results
CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
EXPECTED_BRANCH=feature/fastmri-brain-adapter
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

acceleration="${ACCELERATION:-4}"
case "$acceleration" in
    4) center_fraction=0.08 ;;
    8) center_fraction=0.04 ;;
    *) echo "ACCELERATION must be 4 or 8" >&2; exit 2 ;;
esac

max_volumes="${MAX_VOLUMES:-10}"
experiment_id="fastmri-brain-adapter-deepinv041-r${acceleration}-v1"
output_dir="$RESULTS_ROOT/$experiment_id"

cd "$REPO_ROOT"
mkdir -p "$RESULTS_ROOT"
current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$EXPECTED_BRANCH" ]]; then
    echo "Expected branch $EXPECTED_BRANCH, found $current_branch" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to run with uncommitted repository changes." >&2
    git status --short >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Cached checkpoint not found; offline job will not download it: $CHECKPOINT" >&2
    exit 2
fi
if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing output: $output_dir" >&2
    exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Branch: $current_branch"
echo "Acceleration: $acceleration"
echo "Center fraction: $center_fraction"
echo "Maximum volumes: $max_volumes"
echo "Offline checkpoint: $CHECKPOINT"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi
"$PYTHON_BIN" -c 'import torch, deepinv, ram.adapters; print("Torch:", torch.__version__); print("DeepInverse:", deepinv.__version__); print("RAM package:", ram.__file__); print("RAM adapters:", ram.adapters.__file__); print("CUDA:", torch.version.cuda)'

"$PYTHON_BIN" scripts/benchmark_fastmri_brain_vcc_ram.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$output_dir" \
    --checkpoint "$CHECKPOINT" \
    --acceleration "$acceleration" \
    --center-fraction "$center_fraction" \
    --normalization-scale 0.005 \
    --noise-sigma 0.0005 \
    --max-volumes "$max_volumes" \
    --slices-per-volume 3 \
    --edge-fraction 0.15 \
    --esc-max-iterations 60 \
    --seed 0 \
    --save-previews 6 \
    --device cuda 2>&1 | tee "$output_dir.run.log"

mv "$output_dir.run.log" "$output_dir/run.log"
echo "Finished: $(date --iso-8601=seconds)"
