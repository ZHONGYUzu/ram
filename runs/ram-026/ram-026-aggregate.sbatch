#!/bin/bash
#SBATCH --job-name=ram-026-summary
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-026/logs/summary-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-026/logs/summary-%j.err

set -euo pipefail
RUN=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-026
BASE=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-025
PYTHON=/home/students/studxuzho1/envs/ram/bin/python
export MPLBACKEND=Agg
test ! -e "$RUN/summary"
"$PYTHON" "$RUN/code/scripts/aggregate_ram026.py" --run-dir "$RUN" --baseline-run-dir "$BASE"
