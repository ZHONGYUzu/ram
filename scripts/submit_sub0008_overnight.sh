#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to submit from a repository with uncommitted changes." >&2
    exit 1
fi

mkdir -p logs

overnight_job="$(sbatch --parsable --job-name=ram_dc_overnight --export=ALL,BUNDLE=all scripts/slurm_sub0008_overnight_dc.sbatch)"

echo "Submitted Sub0008 overnight job: $overnight_job"
echo "The job will run up to 168 experiments sequentially."
echo "Resubmitting after timeout is safe; completed experiments are skipped."
echo "Monitor with: squeue -u \"$USER\""
