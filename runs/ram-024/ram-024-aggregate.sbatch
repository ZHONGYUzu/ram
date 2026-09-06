#!/bin/bash
#SBATCH --job-name=ram-024-summary
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-024/logs/summary-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-024/logs/summary-%j.err

set -euo pipefail

RUN=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-024
PYTHON=/home/students/studxuzho1/envs/ram/bin/python
export MPLBACKEND=Agg

test -x "$PYTHON"
test ! -e "$RUN/summary"
"$PYTHON" "$RUN/code/scripts/aggregate_ram024.py" --run-dir "$RUN"
