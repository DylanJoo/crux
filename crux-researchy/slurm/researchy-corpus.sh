#!/bin/sh
#SBATCH --job-name=collect-cw22
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-10
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate crux
cd ~/crux-scale/crux-researchy/

python researchy-corpus.py --shard $SLURM_ARRAY_TASK_ID --num_shards 10
