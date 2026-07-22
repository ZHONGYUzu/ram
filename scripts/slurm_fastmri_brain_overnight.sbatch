#!/bin/bash

# Runs a bounded, sequential fastMRI brain parameter sweep. Set BUNDLE to one
# of all, core, scale, noise, or mask to run only one family.

#SBATCH --job-name=ram_fmri_brain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/students/studxuzho1/ram
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
ESPIRIT_H5=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val_espirit/file_brain_AXFLAIR_200_6002462.h5
RAW_H5=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val/file_brain_AXFLAIR_200_6002462.h5
RESULTS_ROOT=/home/students/studxuzho1/ram-results
SUMMARY_CSV=$RESULTS_ROOT/fastmri-brain-overnight-sweep.csv

bundle="${BUNDLE:-all}"
completed=0
failed=0
skipped=0

finish() {
    exit_status=$?
    echo "Completed this invocation: $completed"
    echo "Failed this invocation: $failed"
    echo "Skipped existing: $skipped"
    echo "Exit status: $exit_status"
    echo "Finished: $(date --iso-8601=seconds)"
    exit "$exit_status"
}
trap finish EXIT

cd "$REPO_ROOT"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
mkdir -p logs "$RESULTS_ROOT"

if [[ ! -x "$PYTHON_BIN" || ! -f "$ESPIRIT_H5" || ! -f "$RAW_H5" ]]; then
    echo "Required Python or fastMRI input is unavailable." >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to run with uncommitted repository changes." >&2
    git status --short >&2
    exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Bundle: $bundle"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Git commit: $(git rev-parse HEAD)"
nvidia-smi
"$PYTHON_BIN" -c 'import torch, deepinv; print("Torch:", torch.__version__); print("CUDA:", torch.version.cuda); print("DeepInverse:", deepinv.__version__); print("CUDA available:", torch.cuda.is_available())'

run_experiment() {
    local experiment_id="$1"
    local reference_key="$2"
    local acceleration="$3"
    local center_fraction="$4"
    local mask_type="$5"
    local normalization_scale="$6"
    local noise_sigma="$7"
    local add_noise="$8"
    local seed="$9"
    local run_dir="$RESULTS_ROOT/$experiment_id"
    local console_log="$RESULTS_ROOT/${experiment_id}-console.log"

    if [[ -f "$run_dir/metrics.json" ]]; then
        echo "Skipping completed experiment: $experiment_id"
        skipped=$((skipped + 1))
        return
    fi
    if [[ -e "$run_dir" ]]; then
        echo "Skipping incomplete existing directory: $run_dir" >&2
        failed=$((failed + 1))
        return
    fi

    command=(
        "$PYTHON_BIN" scripts/validate_fastmri_brain_ram.py
        --input-h5 "$ESPIRIT_H5"
        --raw-h5 "$RAW_H5"
        --output-dir "$run_dir"
        --reference-key "$reference_key"
        --map-index auto
        --slices 7 8 9
        --acceleration "$acceleration"
        --center-fraction "$center_fraction"
        --mask-type "$mask_type"
        --normalization-scale "$normalization_scale"
        --noise-sigma "$noise_sigma"
        --seed "$seed"
        --device cuda
    )
    if [[ "$add_noise" == "true" ]]; then
        command+=(--add-noise)
    else
        command+=(--no-add-noise)
    fi

    echo "Starting: $experiment_id"
    printf 'Command: '
    printf '%q ' "${command[@]}"
    printf '\n'
    if "${command[@]}" 2>&1 | tee "$console_log"; then
        mv "$console_log" "$run_dir/run.log"
        completed=$((completed + 1))
        echo "Completed: $experiment_id"
    else
        status=$?
        if [[ -d "$run_dir" ]]; then
            mv "$console_log" "$run_dir/run.log"
        fi
        failed=$((failed + 1))
        echo "Failed ($status): $experiment_id" >&2
    fi
}

slug() {
    local value="$1"
    echo "${value//./p}"
}

run_core() {
    for seed in 0 1 2; do
        run_experiment "fastmri-brain-sweep-paper-r8-seed${seed}" reference_acl15 8 0.04 random 0.005 0.0005 true "$seed"
        run_experiment "fastmri-brain-sweep-paper-r4-seed${seed}" reference_acl15 4 0.08 random 0.005 0.0005 true "$seed"
    done
}

run_scale() {
    for scale in 0.0002 0.0005 0.001 0.0025 0.01 0.02; do
        run_experiment "fastmri-brain-sweep-scale-$(slug "$scale")" reference_acl15 8 0.04 random "$scale" 0.0005 true 0
    done
}

run_noise() {
    for sigma in 0.0001 0.00025 0.001 0.002; do
        run_experiment "fastmri-brain-sweep-noisy-sigma-$(slug "$sigma")" reference_acl15 8 0.04 random 0.005 "$sigma" true 0
    done
    for sigma in 0.0001 0.0005 0.001 0.002; do
        run_experiment "fastmri-brain-sweep-noiseless-cond-$(slug "$sigma")" reference_acl15 8 0.04 random 0.005 "$sigma" false 0
    done
}

run_mask() {
    for seed in 0 1; do
        run_experiment "fastmri-brain-sweep-equispaced-r8-seed${seed}" reference_acl15 8 0.04 equispaced 0.005 0.0005 true "$seed"
    done
    run_experiment fastmri-brain-sweep-center-0p02 reference_acl15 8 0.02 random 0.005 0.0005 true 0
    run_experiment fastmri-brain-sweep-center-0p08 reference_acl15 8 0.08 random 0.005 0.0005 true 0
    run_experiment fastmri-brain-sweep-acl30 reference_acl30 8 0.04 random 0.005 0.0005 true 0
}

case "$bundle" in
    all)
        run_core
        run_scale
        run_noise
        run_mask
        ;;
    core) run_core ;;
    scale) run_scale ;;
    noise) run_noise ;;
    mask) run_mask ;;
    *)
        echo "Unknown BUNDLE: $bundle" >&2
        exit 2
        ;;
esac

"$PYTHON_BIN" scripts/summarize_fastmri_brain_sweep.py \
    --results-root "$RESULTS_ROOT" \
    --experiment-glob 'fastmri-brain-sweep-*' \
    --output-csv "$SUMMARY_CSV"

if (( failed > 0 )); then
    exit 1
fi
