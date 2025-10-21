#!/bin/sh
#SBATCH --job-name=encode-d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

module load anaconda3/2024.2
conda activate crc

for SHARD in {0..1};do
    CUDA_VISIBLE_DEVICES=$SHARD python -m tevatron.retriever.driver.encode  \
      --output_dir=temp \
      --model_name_or_path Qwen/Qwen3-Embedding-8B \
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
