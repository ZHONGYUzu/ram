#!/bin/bash

# ram-020: formal AXT1POST simulated 15-coil R8 OOD benchmark.

#SBATCH --job-name=ram020_mc15_t1post
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-020/logs/slurm-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-020/logs/slurm-%j.err

cd /home/students/studxuzho1/ram-fastmri-brain-adapter
exec bash scripts/run_fastmri_brain_formal_multicoil.sh \
    ram-020 AXT1POST axt1post-simulated-multicoil-r8 \
    file_brain_AXT1POST_200_6001976.h5 \
    file_brain_AXT1POST_200_6001979.h5 \
    file_brain_AXT1POST_200_6001982.h5 \
    file_brain_AXT1POST_200_6001988.h5 \
    file_brain_AXT1POST_200_6002016.h5
