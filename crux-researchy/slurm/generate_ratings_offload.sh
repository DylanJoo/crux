#!/bin/sh
#SBATCH --job-name=crux-out
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-10%10
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
module load anaconda3/2024.2
conda activate crux

echo Running shard $SLURM_ARRAY_TASK_ID
root_dir=/exp/scale25/artifacts/crux

# Generate ratings for each researchy questions with top 20

# Training examples with init q 
# python3 -m crux.augmentation.gen_ratings_offload \
#     --config $HOME/crux/configs/default_config.yaml \
#     --dataset researchy \
#     --corpus $root_dir/crux-researchy/docs/cw22-b-researchy-v1/corpus.pkl \
#     --output_dir $root_dir/crux-researchy/judge-offload \
#     --run_path $root_dir/crux-researchy/runs/run.researchy-init-q.bm25+qwen3.clueweb22-b.txt \
#     --top_k 20 \
#     --shard $SLURM_ARRAY_TASK_ID --total_shards 100 \
#     --batch_size 64

# Testing examples with init q 
python3 -m crux.augmentation.gen_ratings_offload \
    --config $HOME/crux/configs/default_config.yaml \
    --dataset researchy \
    --corpus $root_dir/crux-researchy/docs/cw22-b.researchy-v1/doc00.pkl \
    --output_dir $root_dir/crux-researchy/judge-offload \
    --run_path $root_dir/crux-researchy/runs/run.researchy-test-init-q.bm25+qwen3.clueweb22-b.txt \
    --top_k 20 \
    --shard $SLURM_ARRAY_TASK_ID --total_shards 10 \
    --batch_size 64
