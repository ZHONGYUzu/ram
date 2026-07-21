#!/bin/bash

# Submit through scripts/submit_sub0008_overnight.sh. Each ordinary Slurm job
# runs a bundle of independent experiments sequentially to stay below the
# account's submitted-job limit.

#SBATCH --job-name=ram_dc_bundle
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/students/studxuzho1/ram
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
INPUT_H5=/mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0008.h5
MASK_TXT=/home/students/studxusiy1/mr_recon/masks/mask_VISTA_132x25_acc8_8.txt
RESULTS_ROOT=/home/students/studxuzho1/ram-results

bundle="${BUNDLE:?Submit with BUNDLE=core, grid-a, grid-b, or grid-c}"

finish() {
    exit_status=$?
    echo "Exit status: $exit_status"
    echo "Finished: $(date --iso-8601=seconds)"
    exit "$exit_status"
}
trap finish EXIT

cd "$REPO_ROOT"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

if [[ ! -x "$PYTHON_BIN" || ! -f "$INPUT_H5" || ! -f "$MASK_TXT" ]]; then
    echo "Required Python, H5, or mask input is unavailable." >&2
    exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Bundle: $bundle"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Git commit: $(git rev-parse HEAD)"

nvidia-smi
"$PYTHON_BIN" -c 'import torch; print("Torch:", torch.__version__); print("CUDA:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'

run_experiment() {
    local experiment_id="$1"
    local slice_index="$2"
    local time_index="$3"
    local gamma="$4"
    local dc_cg_iter="$5"

    if [[ -e "$RESULTS_ROOT/$experiment_id" ]]; then
        echo "Skipping existing experiment: $experiment_id"
        return
    fi

    echo "Starting experiment: $experiment_id"
    echo "Slice/time: $slice_index/$time_index"
    echo "DC gamma/iterations: $gamma/$dc_cg_iter"

    bash scripts/run_cine_experiment.sh \
        --experiment-id "$experiment_id" \
        --input-h5 "$INPUT_H5" \
        --mask-txt "$MASK_TXT" \
        --results-root "$RESULTS_ROOT" \
        --slice-index "$slice_index" \
        --time-index "$time_index" \
        --normalization p99.5 \
        --dc-gamma "$gamma" \
        --dc-cg-iter "$dc_cg_iter" \
        --batch-size 1 \
        --device cuda
}

gammas=(0 10 30 100)
times=(2 7 12 17 22)

case "$bundle" in
    core)
        for dc_cg_iter in 8 16 32; do
            for gamma in 3 10 30 100; do
                run_experiment "sub0008-refine-s06-t10-g${gamma}-cg${dc_cg_iter}" 6 10 "$gamma" "$dc_cg_iter"
            done
        done

        for time_index in 0 3 6 9 12 15 18 21 24; do
            for gamma in "${gammas[@]}"; do
                run_experiment "sub0008-temporal-s06-t$(printf '%02d' "$time_index")-g${gamma}-cg32" 6 "$time_index" "$gamma" 32
            done
        done
        ;;
    grid-a) slices=(1 3) ;;
    grid-b) slices=(5 7) ;;
    grid-c) slices=(9 11) ;;
    *)
        echo "Unknown BUNDLE: $bundle" >&2
        exit 2
        ;;
esac

if [[ "$bundle" == grid-* ]]; then
    for slice_index in "${slices[@]}"; do
        for time_index in "${times[@]}"; do
            for gamma in "${gammas[@]}"; do
                run_experiment "sub0008-grid-s$(printf '%02d' "$slice_index")-t$(printf '%02d' "$time_index")-g${gamma}-cg32" "$slice_index" "$time_index" "$gamma" 32
            done
        done
    done
fi
