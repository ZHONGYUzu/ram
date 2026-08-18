#!/bin/bash

# ram-021: formal AXT1PRE simulated 15-coil R8 OOD benchmark.

#SBATCH --job-name=ram021_mc15_t1pre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-021/logs/slurm-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-021/logs/slurm-%j.err

cd /home/students/studxuzho1/ram-fastmri-brain-adapter
exec bash scripts/run_fastmri_brain_formal_multicoil.sh \
    ram-021 AXT1PRE axt1pre-simulated-multicoil-r8 \
    file_brain_AXT1PRE_200_6002079.h5 \
    file_brain_AXT1PRE_200_6002087.h5 \
    file_brain_AXT1PRE_200_6002101.h5 \
    file_brain_AXT1PRE_200_6002108.h5 \
    file_brain_AXT1PRE_200_6002120.h5
