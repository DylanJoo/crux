#!/bin/sh
#SBATCH --job-name=qwen3-0.6B
#SBATCH --cpus-per-task=32
#SBATCH --partition gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=4,9,10
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
module load anaconda3/2024.2
conda activate crux

cd ~/crux/crux-researchy/

# python rerank-with-subquestions.py \
#     --batch_size 100 \
#     --shard $SLURM_ARRAY_TASK_ID \
#     --total_shards 10 \
#     --split train

python rerank-with-subquestions.py \
    --batch_size 100  \
    --shard $SLURM_ARRAY_TASK_ID --total_shards 10  \
    --split test
