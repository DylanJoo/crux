#!/bin/bash -l
#SBATCH --job-name=vllm-api
#SBATCH --partition=dev-g                    # partition name
#SBATCH --nodes=1                            # Total number of nodes 
#SBATCH --ntasks-per-node=1                  # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=2                    # Allocate one gpu per MPI rank
#SBATCH --time=00:10:00                      # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001396          # Project for billing
#SBATCH --output=logs/%x.out 
#SBATCH --error=logs/%x.err 

source ${HOME}/.bashrc
cd ~/crux

SIF=~/images/rocm-vllm_ubuntu22.04_rocm6.3.1_py3.11_torch2.6.0_vllm_01-20-2025.sif
export SINGULARITY_BIND="/scratch/project_465001396,/scratch/project_465001640"

# nodes=1
# gpus-per-node=4
singularity exec $SIF python3 vllm_api.py
