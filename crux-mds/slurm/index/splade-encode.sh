#!/bin/sh
#SBATCH --job-name=lucene
#SBATCH --partition gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out


# Set-up the environment.
module load anaconda3/2024.2
conda activate crux

python3 -m crux.retrieve.mlm_encode \
    --model_name_or_path naver/splade-v3 \
    --tokenizer_name naver/splade-v3 \
    --collection ${HOME}/datasets/crux-mds-corpus/collections \
    --collection_output ${HOME}/datasets/crux-mds-corpus/indices/splade-v3.crux-mds-corpus.lucene/vectors.jsonl \
    --batch_size 128 \
    --max_length 512 \
    --quantization_factor 100
