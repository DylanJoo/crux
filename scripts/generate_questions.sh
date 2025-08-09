#!/bin/sh
#SBATCH --job-name=questions
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
mkdir $output_dir

# Start the experiment.
python3 -m augmentation.gen_questions \
    --config configs/scale.litellm.yaml \
    --input configs/scale.litellm.yaml \
    --output_dir $output_dir \
    --dataset_name neuclir_mt \
    --split train \
    --tag ques-gen \
    --load_mode litellm \
    --max_new_tokens 4096
