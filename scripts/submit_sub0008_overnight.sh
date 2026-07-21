#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to submit from a repository with uncommitted changes." >&2
    exit 1
fi

mkdir -p logs

core_job="$(sbatch --parsable --job-name=ram_dc_core --export=ALL,BUNDLE=core scripts/slurm_sub0008_overnight_dc.sbatch)"
grid_a_job="$(sbatch --parsable --job-name=ram_dc_grid_a --export=ALL,BUNDLE=grid-a scripts/slurm_sub0008_overnight_dc.sbatch)"
grid_b_job="$(sbatch --parsable --job-name=ram_dc_grid_b --export=ALL,BUNDLE=grid-b scripts/slurm_sub0008_overnight_dc.sbatch)"
grid_c_job="$(sbatch --parsable --job-name=ram_dc_grid_c --export=ALL,BUNDLE=grid-c scripts/slurm_sub0008_overnight_dc.sbatch)"

echo "Submitted Sub0008 overnight campaign (168 experiments in four jobs)."
echo "Core job:   $core_job (48 experiments)"
echo "Grid A job: $grid_a_job (40 experiments; slices 1 and 3)"
echo "Grid B job: $grid_b_job (40 experiments; slices 5 and 7)"
echo "Grid C job: $grid_c_job (40 experiments; slices 9 and 11)"
echo "Monitor with: squeue -u \"$USER\""
