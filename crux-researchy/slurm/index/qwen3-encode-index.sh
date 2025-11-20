#!/bin/bash -l
#SBATCH --job-name=encode-d
#SBATCH --output=enc-doc.out.%a
#SBATCH --error=enc-doc.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0,64
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate pyserini

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
