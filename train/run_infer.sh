#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=infer
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag

# python3 infer_standard.py
python3 infer_standard_with_prefix.py

