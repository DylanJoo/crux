#!/bin/sh
#SBATCH --job-name=qwen3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --array=1-10%2
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Multi-GPU encoding
MULTIJOBS=${HOME}/multigpu.txt
source ~/.bashrc
enter_conda
conda activate crc

mkdir -p ${HOME}/temp/output_qwen3
N_SHARD=10

each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
echo It is running on $each task

for lang in zho; do
python -m tevatron.retriever.driver.encode  \
    --output_dir=temp \
    --model_name_or_path Qwen/Qwen3-Embedding-4B \
    --bf16 \
    --per_device_eval_batch_size 128 \
    --normalize \
    --pooling last \
    --padding_side left \
    --passage_prefix "" \
    --passage_max_len 512 \
    --dataset_name scale/neuclir \
    --dataset_path /exp/scale25/neuclir/docs/${lang}.mt.jsonl \
    --dataset_split none \
    --dataset_number_of_shards ${N_SHARD} \
    --encode_output_path /home/hltcoe/jhueiju/temp/output_qwen3/corpus_neuclir_${lang}.mt.pkl$each \
    --dataset_shard_index $each
done
