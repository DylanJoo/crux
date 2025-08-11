#!/bin/bash -l
#SBATCH --job-name=api              # Job name
#SBATCH --output=logs/api.o%j       # Name of stdout output file
#SBATCH --error=logs/api.e%j        # Name of stderr error file
#SBATCH --partition=standard-g         # partition name
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --ntasks-per-node=8         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=8           # Allocate one gpu per MPI rank
#SBATCH --time=2-00:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001396 # Project for billing

source ${HOME}/.bashrc
cd ~/mdrag
# 8591756 small-g
# 8591396 standard-g
# 8591897 standard-g

conda activate base

SIF=/scratch/project_465001396/images/rocm-vllm_ubuntu22.04_rocm6.2_py3.10_torch2.3.0_vllm0.5.5.sif

# export ROCR_VISIBLE_DEVICES=0,1,2,3
singularity exec --bind /scratch/project_465001396 ${SIF} \
    vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
        --dtype bfloat16 --pipeline-parallel-size 8
