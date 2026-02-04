#!/bin/sh
#SBATCH --job-name=lucene
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

## Use the download collection from: https://huggingface.co/datasets/DylanJHJ/crux-mds-corpus
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input ${HOME}/datasets/crux-mds-corpus/collections \
    --index ${HOME}/datasets/crux-mds-corpus/indices/bm25.crux.crux-mds-corpus.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 128
