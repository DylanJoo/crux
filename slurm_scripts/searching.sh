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
source ~/.bashrc
enter_conda
conda activate crux
cd ~/crux-scale/preprocessing_scripts/

python researchy-run.py --shard $SLURM_ARRAY_TASK_ID --num_shards 10
