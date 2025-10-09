#!/bin/sh
#SBATCH --job-name=collect-cw22
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-20
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
module load anaconda3/2024.2
conda activate crux
cd ~/crux/crux-researchy/

# CW-22-research v1 (trainin init query)
# python researchy-corpus.py --shard $SLURM_ARRAY_TASK_ID --num_shards 10 \
#     --input_run /exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-init-q_bm25.clueweb22-b.txt \
#     --output_corpus /exp/scale25/artifacts/crux/crux-researchy/docs/cw22-b.researchy-v1/doc00.jsonl

# CW-22-research v2 (trainin gpt4 query)
# python3 create-researchy-corpus.py --shard $SLURM_ARRAY_TASK_ID --num_shards 10 \
#     --input_run /exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-gpt4-q_bm25.clueweb22-b.txt \
#     --output_corpus /exp/scale25/artifacts/crux/crux-researchy/docs/cw22-b.researchy-v2/doc00.jsonl

# Updated CW-22-research v1 (trainin init query + testing init query)
python3 create-researchy-corpus.py --shard $SLURM_ARRAY_TASK_ID --num_shards 20 \
    --input_run /exp/scale25/artifacts/crux/crux-researchy/runs/run.all.init-q.txt \
    --output_corpus /exp/scale25/artifacts/crux/crux-researchy/docs/cw22-b.researchy-v1/doc00.jsonl
