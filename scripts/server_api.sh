#!/bin/bash -l
#SBATCH --job-name=s-psg            # Job name
#SBATCH --output=logs/api.o%j   # Name of stdout output file
#SBATCH --error=logs/api.e%j    # Name of stderr error file
#SBATCH --partition=small-g         # partition name
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --ntasks-per-node=4         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --time=3-00:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001339 # Project for billing

source ${HOME}/.bashrc
cd ~/mdrag

# SIF=~/temp/rocm-vllm_ubuntu20.04_rocm6.1.2_py3.9_vllm0.5.5.sif
SIF=~/temp/rocm-vllm_ubuntu22.04_rocm6.2_py3.10_torch2.3.0_vllm0.5.5.sif
# SIF=~/temp/rocm-vllm_ubuntu22.04_rocm6.2_py3.10_torch2.5.0_09-07-2024_vllm0.5.5.sif


# export ROCR_VISIBLE_DEVICES=0,1,2,3
singularity exec --bind /scratch/project_465001339 ${SIF} \
    vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
        --dtype bfloat16 --pipeline-parallel-size 4 

# export ROCR_VISIBLE_DEVICES=4,5,6,7
# singularity exec --bind /scratch/project_465001339 ${SIF} \
#     vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
#         --dtype bfloat16 --pipeline-parallel-size 4 --port 8001
