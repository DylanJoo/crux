#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=bm25
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

# index
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input ${DATASET_DIR}/RACE/passages \
    --index ${INDEX_DIR}/RACE/bm25.race.passages.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 16

# search
for split in test testb;do
    python -m pyserini.search.lucene \
      --index ${INDEX_DIR}/RACE/bm25.race.passages.lucene \
      --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
      --output runs/baseline.bm25.race-${split}.passages.run \
      --hits 100 \
      --bm25
done
