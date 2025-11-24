#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=lucene
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out


# Set-up the environment.
module load anaconda3/2024.2
conda activate crux

python -m pyserini.index.lucene \
    --collection JsonVectorCollection \
    --input ${HOME}/datasets/crux-mds-corpus/indices/splade-v3.crux-mds-corpus.lucene \
    --index ${HOME}/datasets/crux-mds-corpus/indices/splade-v3.crux-mds-corpus.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 32 \
    --impact --pretokenized
