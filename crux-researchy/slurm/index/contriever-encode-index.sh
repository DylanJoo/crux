#!/bin/sh
#SBATCH --job-name=encode-d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
module load anaconda3/2024.2
conda activate crux

python -m pyserini.encode \
    input   --corpus ${HOME}/datasets/crux-mds-corpus/collections \
            --fields text \
            --delimiter "__IMPOSSIBLE_DELIMITER__" \
    output  --embeddings ${HOME}/datasets/crux-mds-corpus/indices/contriever-msmarco.crux-mds-corpus.faiss \
            --to-faiss \
    encoder --encoder facebook/contriever-msmarco \
            --encoder-class contriever \
            --fields text \
            --max-length 384 \
            --batch 64 \
            --fp16
