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

# duc04
python -m tevatron.retriever.driver.search \
    --query_reps ${CRUX_CORPUS_ROOT}/indices/qwen3-embedding-8b.crux-mds-duc04.queries.pkl \
    --passage_reps "${CRUX_CORPUS_ROOT}/indices/qwen3-embedding-8b.crux-mds-corpus.pkl/*.pkl" \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${HOME}/temp/tevatron.txt 

python -m tevatron.utils.format.convert_result_to_trec \
    --input ${HOME}/temp/tevatron.txt \
    --output ${HOME}/crux/runs/qwen3-embedding-8b.crux-mds-duc04.txt \
    --remove_query

# multi-news
python -m tevatron.retriever.driver.search \
    --query_reps ${CRUX_CORPUS_ROOT}/indices/qwen3-embedding-8b.crux-mds-multi_news.queries.pkl \
    --passage_reps "${CRUX_CORPUS_ROOT}/indices/qwen3-embedding-8b.crux-mds-corpus.pkl/*.pkl" \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${HOME}/temp/tevatron.txt 

python -m tevatron.utils.format.convert_result_to_trec \
    --input ${HOME}/temp/tevatron.txt \
    --output ${HOME}/crux/runs/qwen3-embedding-8b.crux-mds-multi_news.txt \
    --remove_query
