#!/bin/sh
#SBATCH --job-name=search
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

module load anaconda3/2024.2
conda activate crux

export CRUX_ROOT=${HOME}/datasets/crux
export CRUX_CORPUS_ROOT=${CRUX_ROOT/crux/crux-mds-corpus}

mkdir -p ${HOME}/crux/runs

TOPIC_PATH=${CRUX_ROOT}/crux-mds-duc04/topic/requests.Llama-3.1-70B-Instruct.0-1.jsonl
python -m crux.tools.generic.convert_jsonl_to_tsv --input $TOPIC_PATH
python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index ${CRUX_CORPUS_ROOT}/indices/splade-v3.crux-mds-corpus.lucene \
    --topics ${TOPIC_PATH/jsonl/tsv} \
    --encoder naver/splade-v3 \
    --output ${HOME}/crux/runs/splade-v3.default.crux-mds-duc04.txt \
    --hits 100 --impact 


# convert jsonl to tsv
TOPIC_PATH=${CRUX_ROOT}/crux-mds-multi_news/topic/requests.Llama-3.1-70B-Instruct.0-1.small.jsonl
python -m crux.tools.generic.convert_jsonl_to_tsv --input $TOPIC_PATH
python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index ${CRUX_CORPUS_ROOT}/indices/splade-v3.crux-mds-corpus.lucene \
    --topics ${TOPIC_PATH/jsonl/tsv} \
    --encoder naver/splade-v3 \
    --output ${HOME}/crux/runs/splade-v3.default.crux-mds-multi_news.txt \
    --hits 100 --impact 
