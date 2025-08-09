#!/bin/sh
#SBATCH --job-name=crux-request
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out

# Load the environment
source $HOME/.bashrc
enter_conda
conda activate crux

# Set the path to the multijobs file
root_dir=/exp/scale25/artifacts/crux

for subset in multi_news duc04;do
python3 -m crux.augmentation.gen_topics \
    --config $HOME/crux-scale/configs/default_config.yaml \
    --dataset mds --subset $subset \
    --output_dir $root_dir/crux-mds-${subset}/topics \
    --max_new_tokens 128 \
    --batch_size 32 \
    --load_mode litellm
done
