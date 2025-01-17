#!/bin/bash -l
#SBATCH --job-name=train-topic-gen           # Job name
#SBATCH --output=logs/train-topic-gen.o%j    # Name of stdout output file
#SBATCH --error=logs/train-topic-gen.e%j     # Name of stderr error file
#SBATCH --partition=dev-g                    # partition name
#SBATCH --nodes=1                            # Total number of nodes 
#SBATCH --ntasks-per-node=1                  # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=1                    # Allocate one gpu per MPI rank
#SBATCH --time=00:30:00                      # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001396          # Project for billing

source ${HOME}/.bashrc
cd ~/crux

dataset_dir=~/datasets
SIF=~/images/rocm-vllm_ubuntu22.04_rocm6.2_py3.10_torch2.3.0_vllm0.5.5.sif
export SINGULARITY_BIND="/scratch/project_465001396,/scratch/project_465001640"

# Start the experiment.
for shard_i in $(seq 0 1);do
    singularity exec $SIF \
    python3 -m augmentation.gen_topics \
        --config configs/crux-testing-8b.yaml \
        --multi_news_file ${dataset_dir}/multi_news \
        --shard $shard_i --shard_size 100 \
        --split train \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --batch_size 1 \
        --tag topics-gen \
        --temperature 0.7 \
        --max_new_tokens 128 \
        --output_dir ${DATASET_DIR}/mdrag/shard_data/
done
