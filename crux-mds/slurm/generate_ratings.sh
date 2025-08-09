#!/bin/sh
#SBATCH --job-name=crux-cw22
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-1000%3
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out

# Load the environment
source $HOME/.bashrc
enter_conda
conda activate crux

# Set the path to the multijobs file
MULTIJOBS=${HOME}/multishards.txt
each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
echo Running on: $each

root_dir=/exp/scale25/artifacts/crux

python3 -m crux.augmentation.gen_ratings \
    --config $HOME/crux-scale/configs/default_config.yaml \
    --dataset researchy \
    --corpus $root_dir/crux-researchy/docs/cw22-b-researchy-v1/corpus.pkl \
    --output_dir $root_dir/crux-researchy/qrel \
    --run_path $root_dir/crux-researchy/runs/run.researchy-init-q.bm25.clueweb22-b.txt \
    ${each} --total_shards 1000 \
    --batch_size 32 \
    --load_mode litellm

