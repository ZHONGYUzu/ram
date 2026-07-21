#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to submit from a repository with uncommitted changes." >&2
    exit 1
fi

mkdir -p logs

refine_job="$(sbatch --parsable --job-name=ram_dc_refine --array=0-11 --export=ALL,CAMPAIGN=refine scripts/slurm_sub0008_overnight_dc.sbatch)"
temporal_job="$(sbatch --parsable --job-name=ram_dc_time --array=0-35 --dependency="afterany:$refine_job" --export=ALL,CAMPAIGN=temporal scripts/slurm_sub0008_overnight_dc.sbatch)"
grid_job="$(sbatch --parsable --job-name=ram_dc_grid --array=0-119 --dependency="afterany:$temporal_job" --export=ALL,CAMPAIGN=grid scripts/slurm_sub0008_overnight_dc.sbatch)"

echo "Submitted Sub0008 overnight campaign (168 experiments)."
echo "Refinement array: $refine_job (12 experiments)"
echo "Temporal array:   $temporal_job (36 experiments; after refinement)"
echo "Slice/time grid:  $grid_job (120 experiments; after temporal)"
echo "Monitor with: squeue -u \"$USER\""
