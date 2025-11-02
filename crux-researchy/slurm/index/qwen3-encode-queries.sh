#!/bin/sh
#SBATCH --job-name=encode-q
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

module load anaconda3/2024.2
conda activate crc

export CRUX_ROOT=${HOME}/datasets/crux
export CRUX_CORPUS_ROOT=${CRUX_ROOT/crux/crux-mds-corpus}

TOPIC_PATH=${CRUX_ROOT}/crux-mds-duc04/topic/requests.Llama-3.1-70B-Instruct.0-1.jsonl
MODEL=Qwen/Qwen3-Embedding-8B
python -m tevatron.retriever.driver.encode  \
    --output_dir=temp \
    --attn_implementation sdpa \
    --model_name_or_path $MODEL \
    --bf16 \
    --per_device_eval_batch_size 16 \
    --normalize \
    --pooling last  \
    --padding_side left \
    --query_prefix "Instruct: Given a report request, retrieve relevant passages that provide context to the report.\nQuery:" \
    --append_eos_token \
    --query_max_len 128 \
    --dataset_path ${TOPIC_PATH} \
    --encode_output_path ${CRUX_CORPUS_ROOT}/indices/qwen3-embedding-8b.crux-mds-duc04.queries.pkl \
    --encode_is_query

TOPIC_PATH=${CRUX_ROOT}/crux-mds-multi_news/topic/requests.Llama-3.1-70B-Instruct.0-1.small.jsonl
python -m tevatron.retriever.driver.encode  \
    --output_dir=temp \
    --attn_implementation sdpa \
    --model_name_or_path $MODEL \
    --bf16 \
    --per_device_eval_batch_size 16 \
    --normalize \
    --pooling last  \
    --padding_side left \
    --query_prefix "Instruct: Given a report request, retrieve relevant passages that provide context to the report.\nQuery:" \
    --append_eos_token \
    --query_max_len 128 \
    --dataset_path ${TOPIC_PATH} \
    --encode_output_path ${CRUX_CORPUS_ROOT}/indices/qwen3-embedding-8b.crux-mds-multi_news.queries.pkl \
    --encode_is_query
