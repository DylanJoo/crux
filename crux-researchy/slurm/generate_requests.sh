#!/bin/sh
#SBATCH --job-name=crux-researchy-topic
#SBATCH --output=logs/crux.out.%a
#SBATCH --error=logs/crux.err.%a
#SBATCH --partition=gpu_a100
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --array=0-1%0
#SBATCH --mem=120G
#SBATCH --time=24:00:00

source /sw/arch/RHEL9/EB_production/2024/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate pyserini

crux_root=/home/dju/temp/datasets/crux
export CRUX_ROOT=/home/dju/temp/datasets/crux

# python3 -m crux.augmentation.gen_requests \
#     --config $HOME/crux/configs/default_config.yaml \
#     --model meta-llama/Llama-3.3-70B-Instruct \
#     --num_gpus 4 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --dataset researchy --split test \
#     --output_dir $crux_root/crux-researchy/topic \
#     --max_new_tokens 128 \
#     --batch_size 64 \
#     --load_mode vllm

python3 -m crux.augmentation.gen_requests \
    --config $HOME/crux/configs/default_config.yaml \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --num_gpus 4 \
    --temperature 0.7 \
    --top_p 0.95 \
    --dataset researchy --split train \
    --output_dir $crux_root/crux-researchy/topic \
    --max_new_tokens 128 \
    --batch_size 64 \
    --load_mode vllm \
    --shard $SLURM_ARRAY_TASK_ID --total_shards 100
