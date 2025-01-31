#!/bin/bash -l
#SBATCH --job-name=train-rating-gen
#SBATCH --partition=small-g                    # partition name
#SBATCH --nodes=1                            # Total number of nodes 
#SBATCH --ntasks-per-node=1                  # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=1                    # Allocate one gpu per MPI rank
#SBATCH --time=72:00:00                      # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001396          # Project for billing
#SBATCH --output=logs/%x.%j.out 
#SBATCH --error=logs/%x.%j.err 

source ${HOME}/.bashrc
cd ~/crux

dataset_dir=~/datasets
SIF=~/images/rocm-vllm_ubuntu22.04_rocm6.3.1_py3.11_torch2.6.0_vllm_01-20-2025.sif
export SINGULARITY_BIND="/scratch/project_465001396,/scratch/project_465001640"

# Start the experiment.
for shard_i in $(seq 9 40);do
    singularity exec $SIF \
    python3 -m augmentation.gen_ratings \
        --config configs/crux-default-8b.yaml \
        --shard $shard_i --shard_size 1000 \
        --split train \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --batch_size 128 \
        --tag ratings-gen \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_new_tokens 5 \
        --shard_dir ${dataset_dir}/crux/shard_data/
done
