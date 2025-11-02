#!/bin/sh
#SBATCH --job-name=lucene
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
module load 2024
conda activate ir

python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input ${HOME}/datasets/crux-mds-corpus/collections \
    --index ${HOME}/datasets/crux-mds-corpus/indices/bm25.crux.crux-mds-corpus.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 128
