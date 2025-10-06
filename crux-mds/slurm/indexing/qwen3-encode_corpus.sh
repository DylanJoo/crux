#!/bin/sh
#SBATCH --job-name=encode-d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40
#SBATCH --nodes=1
#SBATCH --array=1-2%2
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

module load anaconda3/2024.2
conda activate crc

MODEL='Qwen/Qwen3-Embedding-8B'
MULTIJOBS=${HOME}/multigpu.txt

N_SHARD=1
each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
echo It is running on $each task

python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path $MODEL \
  --bf16 \
  --per_device_eval_batch_size 128 \
  --normalize \
  --pooling last  \
  --padding_side left \
  --passage_prefix "" \
  --passage_max_len 384 \
  --dataset_config none \
  --dataset_name crux-mds \
  --dataset_split none \
  --dataset_path /exp/scale25/artifacts/crux/temp/passages/testb_psgs.jsonl \
  --dataset_number_of_shards ${N_SHARD} \
  --encode_output_path /exp/scale25/artifacts/crux/temp/qwen3.crux-mds-duc04.passages.tevatron${each} \
  --dataset_shard_index $each

N_SHARD=2
  # --dataset_path "/exp/scale25/artifacts/crux/temp/passages/*.jsonl" \
