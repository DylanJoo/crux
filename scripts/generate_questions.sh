#!/bin/bash -l
#SBATCH --job-name=train-ques-gen
#SBATCH --partition=dev-g                    # partition name
#SBATCH --nodes=1                            # Total number of nodes 
#SBATCH --ntasks-per-node=1                  # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=1                    # Allocate one gpu per MPI rank
#SBATCH --time=00:30:00                      # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001396          # Project for billing
#SBATCH --output=logs/%x.%j.out 
#SBATCH --error=logs/%x.%j.err 

source ${HOME}/.bashrc
cd ~/crux

dataset_dir=~/datasets
SIF=~/images/rocm-vllm_ubuntu22.04_rocm6.3.1_py3.11_torch2.6.0_vllm_01-20-2025.sif
export SINGULARITY_BIND="/scratch/project_465001396,/scratch/project_465001640"

# Start the experiment.
for shard_i in $(seq 0 1);do
    singularity exec $SIF \
    python3 -m augmentation.gen_questions \
        --config configs/crux-default-8b.yaml \
        --multi_news_file ${dataset_dir}/multi_news \
        --shard $shard_i --shard_size 10 \
        --split train \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --batch_size 64 \
        --tag ques-gen \
        --temperature 0.7 \
        --top_p 0.95 \
        --max_new_tokens 640 \
        --output_dir ${dataset_dir}/crux/shard_data/
done
