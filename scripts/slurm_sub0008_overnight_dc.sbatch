#!/bin/bash

# Submit through scripts/submit_sub0008_overnight.sh. The submission selects the
# campaign and matching array bounds.

#SBATCH --job-name=ram_overnight
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/students/studxuzho1/ram
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
INPUT_H5=/mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0008.h5
MASK_TXT=/home/students/studxusiy1/mr_recon/masks/mask_VISTA_132x25_acc8_8.txt
RESULTS_ROOT=/home/students/studxuzho1/ram-results

campaign="${CAMPAIGN:?Submit with CAMPAIGN=refine, temporal, or grid}"
task_id="${SLURM_ARRAY_TASK_ID:?This job must be submitted as an array}"

case "$campaign" in
    refine)
        gammas=(3 10 30 100)
        dc_iters=(8 16 32)
        gamma="${gammas[$((task_id % ${#gammas[@]}))]}"
        dc_cg_iter="${dc_iters[$((task_id / ${#gammas[@]}))]}"
        slice_index=6
        time_index=10
        experiment_id="sub0008-refine-s06-t10-g${gamma}-cg${dc_cg_iter}"
        ;;
    temporal)
        times=(0 3 6 9 12 15 18 21 24)
        gammas=(0 10 30 100)
        gamma="${gammas[$((task_id % ${#gammas[@]}))]}"
        time_index="${times[$((task_id / ${#gammas[@]}))]}"
        dc_cg_iter=32
        slice_index=6
        experiment_id="sub0008-temporal-s06-t$(printf '%02d' "$time_index")-g${gamma}-cg${dc_cg_iter}"
        ;;
    grid)
        slices=(1 3 5 7 9 11)
        times=(2 7 12 17 22)
        gammas=(0 10 30 100)
        gamma_count="${#gammas[@]}"
        time_count="${#times[@]}"
        gamma="${gammas[$((task_id % gamma_count))]}"
        time_slot="$(((task_id / gamma_count) % time_count))"
        slice_slot="$((task_id / (gamma_count * time_count)))"
        time_index="${times[$time_slot]}"
        slice_index="${slices[$slice_slot]}"
        dc_cg_iter=32
        experiment_id="sub0008-grid-s$(printf '%02d' "$slice_index")-t$(printf '%02d' "$time_index")-g${gamma}-cg${dc_cg_iter}"
        ;;
    *)
        echo "Unknown CAMPAIGN: $campaign" >&2
        exit 2
        ;;
esac

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
echo "Array task: $task_id"
echo "Campaign: $campaign"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Experiment: $experiment_id"
echo "Slice/time: $slice_index/$time_index"
echo "DC gamma/iterations: $gamma/$dc_cg_iter"

nvidia-smi
"$PYTHON_BIN" -c 'import torch; print("Torch:", torch.__version__); print("CUDA:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'

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
