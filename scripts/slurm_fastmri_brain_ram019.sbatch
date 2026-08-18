#!/bin/bash

# ram-019: formal AXT1 simulated 15-coil R8 OOD benchmark.

#SBATCH --job-name=ram019_mc15_t1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-019/logs/slurm-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-019/logs/slurm-%j.err

cd /home/students/studxuzho1/ram-fastmri-brain-adapter
exec bash scripts/run_fastmri_brain_formal_multicoil.sh \
    ram-019 AXT1 axt1-simulated-multicoil-r8 \
    file_brain_AXT1_201_6002725.h5 \
    file_brain_AXT1_201_6002736.h5 \
    file_brain_AXT1_201_6002740.h5 \
    file_brain_AXT1_201_6002759.h5 \
    file_brain_AXT1_201_6002786.h5
