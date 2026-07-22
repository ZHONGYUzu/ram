#!/bin/bash

#SBATCH --job-name=ram_strong_dc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ram_strong_dc_%j.out
#SBATCH --error=logs/ram_strong_dc_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/students/studxuzho1/ram
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
INPUT_H5=/mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0008.h5
MASK_TXT=/home/students/studxusiy1/mr_recon/masks/mask_VISTA_132x25_acc8_8.txt
RESULTS_ROOT=/home/students/studxuzho1/ram-results

# Original diagnostic plus representative easy/hard temporal and spatial cases.
FRAMES=(6:10 6:3 6:6 6:18 1:7 3:7)
DC_GAMMAS=(100 300 1000)
DC_CG_ITERS=(8 32)

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
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Planned experiments: 36"

nvidia-smi
"$PYTHON_BIN" -c 'import torch; print("Torch:", torch.__version__); print("CUDA:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'

for frame in "${FRAMES[@]}"; do
    IFS=: read -r slice_index time_index <<< "$frame"
    for dc_cg_iter in "${DC_CG_ITERS[@]}"; do
        for dc_gamma in "${DC_GAMMAS[@]}"; do
            experiment_id="sub0008-strongdc-s$(printf '%02d' "$slice_index")-t$(printf '%02d' "$time_index")-g${dc_gamma}-cg${dc_cg_iter}"

            if [[ -e "$RESULTS_ROOT/$experiment_id" ]]; then
                echo "Skipping existing experiment: $experiment_id"
                continue
            fi

            echo "Starting experiment: $experiment_id"
            bash scripts/run_cine_experiment.sh \
                --experiment-id "$experiment_id" \
                --input-h5 "$INPUT_H5" \
                --mask-txt "$MASK_TXT" \
                --results-root "$RESULTS_ROOT" \
                --slice-index "$slice_index" \
                --time-index "$time_index" \
                --normalization p99.5 \
                --dc-gamma "$dc_gamma" \
                --dc-cg-iter "$dc_cg_iter" \
                --batch-size 1 \
                --device cuda
        done
    done
done
