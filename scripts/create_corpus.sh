#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=create-context
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
cd ~/crux

dataset_dir=~/datasets

for split in train test testb;do
python3 augmentation/create_corpus.py \
    --shard_dir ${dataset_dir}/crux/shard_data \
    --ratings_dir ${dataset_dir}/crux/shard_data/ratings-gen \
    --split ${split} \
    --output_dir ${dataset_dir}/crux
done
