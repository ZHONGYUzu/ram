#!/bin/bash

# Submit once with ACCELERATION=4 and once with ACCELERATION=8.

#SBATCH --job-name=ram_fmri_vcc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/students/studxuzho1/ram/logs/%x_%j.out
#SBATCH --error=/home/students/studxuzho1/ram/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/students/studxuzho1/ram
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
DATA_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
RESULTS_ROOT=/home/students/studxuzho1/ram-results

acceleration="${ACCELERATION:?Submit with ACCELERATION=4 or ACCELERATION=8}"
case "$acceleration" in
    4) center_fraction=0.08 ;;
    8) center_fraction=0.04 ;;
    *) echo "ACCELERATION must be 4 or 8" >&2; exit 2 ;;
esac

max_volumes="${MAX_VOLUMES:-10}"
experiment_id="fastmri-brain-vcc-paper-r${acceleration}-pilot-v1"
output_dir="$RESULTS_ROOT/$experiment_id"

cd "$REPO_ROOT"
mkdir -p logs "$RESULTS_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to run with uncommitted repository changes." >&2
    git status --short >&2
    exit 2
fi
if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing output: $output_dir" >&2
    exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Acceleration: $acceleration"
echo "Center fraction: $center_fraction"
echo "Maximum volumes: $max_volumes"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi
"$PYTHON_BIN" -c 'import torch, deepinv, scipy; print("Torch:", torch.__version__); print("DeepInverse:", deepinv.__version__); print("SciPy:", scipy.__version__); print("CUDA:", torch.version.cuda)'

"$PYTHON_BIN" scripts/benchmark_fastmri_brain_vcc_ram.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$output_dir" \
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
