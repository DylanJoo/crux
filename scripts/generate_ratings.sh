#!/bin/sh
#SBATCH --job-name=rating
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

source $HOME/.bashrc
cd ~/crux-scale

output_dir=$HOME/datasets/
mkdir -p $output_dir

# Start the experiment.
python3 -m augmentation.gen_ratings \
    --config configs/scale.litellm.yaml \
    --dataset_name neuclir \
    --tag crux-human \
    --input /exp/scale25/neuclir/eval/nuggets \ # or someone's generated sub-questions
    --qrels data/neuclir24-all-request.qrel \ # or the new retrieval run result
    --output_dir $output_dir \
    --split train \
    --load_mode litellm \
    --max_new_tokens 5 \
