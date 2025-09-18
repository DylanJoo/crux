#!/bin/sh
#SBATCH --job-name=debug-cw22
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-10
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
module load anaconda3/2024.2
conda activate crux
cd ~/crux/crux-researchy/

# python researchy-corpus.py --shard $SLURM_ARRAY_TASK_ID --num_shards 10

python3 researchy-corpus.py \
    --shard $SLURM_ARRAY_TASK_ID --num_shards 10 \
    --input_run /exp/scale25/artifacts/crux/crux-researchy/runs/debug.run \
    --output_corpus /exp/scale25/artifacts/crux/crux-researchy/docs/qrel/doc00.jsonl
