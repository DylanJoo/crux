#!/bin/sh
#SBATCH --job-name=encode-d
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=2-9%2
#SBATCH --time=24:00:00
#SBATCH --output=enc-doc.out.%a
#SBATCH --error=enc-doc.err.%a

module load 2024
module load Miniconda3/24.7.1-0
module load CUDA/12.6.0/

source /home/jju/temp/miniconda3/etc/profile.d/conda.sh
conda activate ir

SHARD_ID=$SLURM_ARRAY_TASK_ID
model=Qwen/Qwen3-Embedding-8B
output_dir=${HOME}/indices/crux-researchy-corpus/${model##*/}
mkdir -p $output_dir

python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path $model \
    --bf16 \
    --exclude_title \
    --per_device_eval_batch_size 12 \
    --normalize \
    --pooling last  \
    --padding_side left \
    --passage_prefix "" \
    --passage_max_len 4096 \
    --dataset_name DylanJHJ/crux-researchy-corpus \
    --encode_output_path ${output_dir}/corpus_emb.${SHARD_ID}.pkl \
    --dataset_number_of_shards 64 \
    --dataset_shard_index ${SHARD_ID}
