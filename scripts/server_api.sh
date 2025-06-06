#!/bin/bash -l
#SBATCH --job-name=server           # Job name
#SBATCH --output=logs/api.o%j       # Name of stdout output file
#SBATCH --error=logs/api.e%j        # Name of stderr error file
#SBATCH --partition=dev-g           # partition name
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --ntasks-per-node=4         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --time=0-03:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001396 # Project for billing

source ${HOME}/.bashrc
cd ~/crux

dataset_dir=~/datasets
SIF=~/images/rocm-vllm_ubuntu22.04_rocm6.3.1_py3.11_torch2.6.0_vllm_01-20-2025.sif
export SINGULARITY_BIND="/scratch/project_465001396,/scratch/project_465001640"

singularity exec ${SIF} \
    vllm serve meta-llama/Llama-3.3-70B-Instruct --dtype bfloat16 --pipeline-parallel-size 4 
