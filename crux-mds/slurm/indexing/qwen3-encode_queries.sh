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

MODEL='Qwen/Qwen3-Embedding-8B'

# python -m tevatron.retriever.driver.encode  \
#     --output_dir=temp \
#     --model_name_or_path $MODEL \
#     --attn_implementation eager \
#     --bf16 \
#     --per_device_eval_batch_size 16 \
#     --normalize \
#     --pooling last  \
#     --padding_side left \
#     --query_prefix "Instruct: Given a report request, retrieve relevant passages that provide context to the report.\nQuery:" \
#     --query_type request \
#     --append_eos_token \
#     --query_max_len 128 \
#     --dataset_name crux-mds \
#     --dataset_path /exp/scale25/artifacts/crux/temp/neuclir_format/topic.mds_duc04.jsonl \
#     --dataset_config none \
#     --dataset_split none \
#     --encode_output_path /exp/scale25/artifacts/crux/temp/qwen3.crux-mds-duc04.queries.tevatron \
#     --encode_is_query

python -m tevatron.retriever.driver.encode  \
    --output_dir=temp \
    --model_name_or_path $MODEL \
    --attn_implementation eager \
    --bf16 \
    --per_device_eval_batch_size 16 \
    --normalize \
    --pooling last  \
    --padding_side left \
    --query_prefix "Instruct: Given a report request, retrieve relevant passages that provide context to the report.\nQuery:" \
    --query_type request \
    --append_eos_token \
    --query_max_len 128 \
    --dataset_name crux-mds \
    --dataset_path /exp/scale25/artifacts/crux/temp/neuclir_format/topic+1sq.mds_duc04.jsonl \
    --dataset_config none \
    --dataset_split none \
    --encode_output_path /exp/scale25/artifacts/crux/temp/qwen3.crux-mds-duc04.queries+1sq.tevatron \
    --encode_is_query

python -m tevatron.retriever.driver.encode  \
    --output_dir=temp \
    --model_name_or_path $MODEL \
    --attn_implementation eager \
    --bf16 \
    --per_device_eval_batch_size 16 \
    --normalize \
    --pooling last  \
    --padding_side left \
    --query_prefix "Instruct: Given a report request, retrieve relevant passages that provide context to the report.\nQuery:" \
    --query_type request \
    --append_eos_token \
    --query_max_len 128 \
    --dataset_name crux-mds \
    --dataset_path /exp/scale25/artifacts/crux/temp/neuclir_format/topic+2sq.mds_duc04.jsonl \
    --dataset_config none \
    --dataset_split none \
    --encode_output_path /exp/scale25/artifacts/crux/temp/qwen3.crux-mds-duc04.queries+2sq.tevatron \
    --encode_is_query
