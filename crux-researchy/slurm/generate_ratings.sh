#!/bin/sh
#SBATCH --job-name=crux-rating
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-10%1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

# Load the environment
source $HOME/.bashrc
enter_conda
conda activate crux

echo Running shard $SLURM_ARRAY_TASK_ID / 100
root_dir=/exp/scale25/artifacts/crux

# Generate ratings for each researchy questions with top 5
python3 -m crux.augmentation.gen_ratings \
    --config $HOME/crux-scale/configs/default_config.yaml \
    --dataset researchy \
    --corpus $root_dir/crux-researchy/docs/cw22-b-researchy-v1/corpus.pkl \
    --output_dir $root_dir/crux-researchy/judge \
    --run_path $root_dir/crux-researchy/runs/run.researchy-init-q.bm25.clueweb22-b.txt \
    --top_k 5 \
    --shard $SLURM_ARRAY_TASK_ID --total_shards 100 \
    --batch_size 64 \
    --load_mode litellm

