#!/bin/sh
#SBATCH --job-name=encode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

model_dir=Qwen/Qwen3-Embedding-0.6B
for SHARD in {0..1};do
    python -m tevatron.retriever.driver.encode  \
        --output_dir=temp \
        --model_name_or_path $model_dir \
        --bf16 \
        --per_device_eval_batch_size 256 \
        --normalize \
        --pooling last  \
        --padding_side left \
        --passage_prefix "" \
        --passage_max_len 384 \
        --dataset_path "/home/hltcoe/jhueiju/datasets/crux-corpus/collections/*.jsonl" \
        --encode_output_path /home/hltcoe/jhueiju/datasets/crux-corpus/indices/qwen3-embedding-8b.crux-mds-corpus.${SHARD}.pkl \
        --dataset_number_of_shards 2 \
        --dataset_shard_index ${SHARD} &
done

wait
echo 'done'
