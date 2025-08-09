#!/bin/sh
#SBATCH --job-name=qwen3-0.6B
#SBATCH --cpus-per-task=32
#SBATCH --partition gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-21%1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate crux
cd ~/crux-scale/crux-researchy

python rerank-candidates.py --batch_size 100 --shard $SLURM_ARRAY_TASK_ID --total_shards 20
