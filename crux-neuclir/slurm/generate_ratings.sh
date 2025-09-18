#!/bin/sh
#SBATCH --job-name=crux-rating
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-0%1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

# Load the environment
source $HOME/.bashrc
enter_conda
conda activate basic

root_dir=/exp/scale25/artifacts/crux

# Generate ratings for each researchy questions with top 5
python3 -m crux.augmentation.gen_ratings \
    --config $HOME/crux-scale/configs/default_config.yaml \
    --dataset neuclir \
    --corpus /exp/scale25/neuclir/docs/mlir.mt.jsonl  \
    --output_dir $root_dir/crux-neuclir/judge/v2 \
    --run_path $root_dir/crux-neuclir/qrel/neuclir24-test-request.qrel \
    --top_k 100 \
    --shard $SLURM_ARRAY_TASK_ID --total_shards 1 \
    --batch_size 64 \
    --load_mode litellm

