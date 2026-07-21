#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_cine_experiment.sh \
    --experiment-id ID \
    --input-h5 PATH \
    --results-root PATH \
    [--mask-mat PATH] \
    [--mask-key KEY] \
    [--slice-index N] \
    [--time-index N] \
    [--batch-size N] \
    [--cg-iter N] \
    [--noise-sigma FLOAT] \
    [--device DEVICE]
EOF
}

experiment_id=""
input_h5=""
results_root=""
mask_mat=""
mask_key="mask"
slice_index=""
time_index=""
batch_size="4"
cg_iter="8"
noise_sigma="1e-3"
device="cuda"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment-id) experiment_id="$2"; shift 2 ;;
        --input-h5) input_h5="$2"; shift 2 ;;
        --results-root) results_root="$2"; shift 2 ;;
        --mask-mat) mask_mat="$2"; shift 2 ;;
        --mask-key) mask_key="$2"; shift 2 ;;
        --slice-index) slice_index="$2"; shift 2 ;;
        --time-index) time_index="$2"; shift 2 ;;
        --batch-size) batch_size="$2"; shift 2 ;;
        --cg-iter) cg_iter="$2"; shift 2 ;;
        --noise-sigma) noise_sigma="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$experiment_id" || -z "$input_h5" || -z "$results_root" ]]; then
    usage >&2
    exit 2
fi

if [[ ! "$experiment_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Experiment ID may contain only letters, numbers, '.', '_', and '-'." >&2
    exit 2
fi

if [[ ! -f "$input_h5" ]]; then
    echo "Input H5 does not exist: $input_h5" >&2
    exit 2
fi

if [[ -n "$mask_mat" && ! -f "$mask_mat" ]]; then
    echo "Mask MAT does not exist: $mask_mat" >&2
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to run with uncommitted repository changes." >&2
    echo "Commit, stash, or intentionally remove the changes first." >&2
    exit 1
fi

run_dir="${results_root%/}/$experiment_id"
if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing experiment directory: $run_dir" >&2
    exit 1
fi

mkdir -p "$run_dir"
output_mat="$run_dir/reconstruction.mat"

command=(
    python scripts/infer_cine_h5.py
    --input-h5 "$input_h5"
    --output-mat "$output_mat"
    --mask-key "$mask_key"
    --batch-size "$batch_size"
    --cg-iter "$cg_iter"
    --noise-sigma "$noise_sigma"
    --device "$device"
)

if [[ -n "$mask_mat" ]]; then
    command+=(--mask-mat "$mask_mat")
fi

if [[ -n "$slice_index" ]]; then
    command+=(--slice-index "$slice_index")
fi

if [[ -n "$time_index" ]]; then
    command+=(--time-index "$time_index")
fi

git rev-parse HEAD > "$run_dir/git-commit.txt"
git status --short --branch > "$run_dir/git-status.txt"
python -m pip freeze > "$run_dir/pip-freeze.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$run_dir/nvidia-smi.txt"
else
    echo "nvidia-smi is unavailable" > "$run_dir/nvidia-smi.txt"
fi

printf '%q ' "${command[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"

echo "Experiment: $experiment_id"
echo "Git commit: $(git rev-parse HEAD)"
echo "Output directory: $run_dir"

"${command[@]}" 2>&1 | tee "$run_dir/run.log"
