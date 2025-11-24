#!/bin/sh
#SBATCH --job-name=bm25-cw22b
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-10
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
module load anaconda3/2024.2
conda activate ir

cd ~/crux/crux-researchy/
#
# init query
python researchy-run.py --q_type init-q --shard $SLURM_ARRAY_TASK_ID --num_shards 2

# GPT4 query
python researchy-run.py --q_type gpt4-q --shard $SLURM_ARRAY_TASK_ID --num_shards 10
