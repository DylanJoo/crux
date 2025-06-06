#!/bin/bash -l
#SBATCH --job-name=vllm-asyn
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
# gpus-per-node=2
singularity exec $SIF python3 vllm_launcher.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --dtype bfloat16 --enforce-eager \
    --tensor-parallel-size 2  --max-model-len 8192

# nodes=1
# gpus-per-node=4
# singularity exec $SIF python3 vllm_launcher.py \
#     --model meta-llama/Llama-3.3-70B-Instruct \
#     --dtype bfloat16 --enforce-eager \
#     --tensor-parallel-size 4  --max-model-len 8192
    
# nodes=1
# gpus-per-node=1
# singularity exec $SIF python3 vllm_launcher.py \
#     --model meta-llama/Llama-3.1-8B-Instruct  \
#     --ftype bfloat16 --enforce-eager \
#     --pipeline-parallel-size 2 --max-model-len 8192
