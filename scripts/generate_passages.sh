#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=psg-gen
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x.%j.out

source ${HOME}/.bashrc
cd ~/mdrag

# Start the experiment.
for shard_i in $(seq 0 40);do
    python3 augmentation/gen_passages.py \
        --shard $shard_i --shard_size 1000 \
        --multi_news_file /home/dju/datasets/multi_news \
        --config configs/mds-decontextualize.llama3-8b.yaml \
        --split train \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --model_tag metallama3.1-8b \
        --tag psgs-gen \
        --load_mode vllm \
        --temperature 0.7 \
        --max_new_tokens 640 \
        --output_dir ${DATASET_DIR}/mdrag/shard_data/ \
        --ampere_gpu 
done
