#!/bin/sh
#SBATCH --job-name=collect-cw22
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-20
#SBATCH --time=24:00:00
#SBATCH --output=%x-%a.out

# Set-up the environment.
module load anaconda3/2024.2
conda activate crux
cd ~/crux/crux-researchy/

rm /exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-all-init-q.bm25.clueweb22-b.txt

# CW-22-research v1 (trainin init query)
cat /exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-*-init-q.bm25.clueweb22-b.txt > /exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-all-init-q.bm25.clueweb22-b.txt
python3 create-researchy-corpus.py --shard $SLURM_ARRAY_TASK_ID --num_shards 10 \
    --input_run /exp/scale25/artifacts/crux/crux-researchy/runs/run.all.txt \
    --output_corpus /exp/scale25/artifacts/crux/crux-researchy/docs/cw22-b.researchy-v1/corpus.jsonl

# CW-22-research v2 (trainin gpt4 query)
# python3 create-researchy-corpus.py --shard $SLURM_ARRAY_TASK_ID --num_shards 10 \
#     --input_run /exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-gpt4-q_bm25.clueweb22-b.txt \
#     --output_corpus /exp/scale25/artifacts/crux/crux-researchy/docs/cw22-b.researchy-v2/doc00.jsonl

