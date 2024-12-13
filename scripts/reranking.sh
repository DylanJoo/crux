#!/bin/sh
#SBATCH --job-name=rerank
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

for retriever in bm25 splade;do
for split in test;do
    model_name=castorini/monot5-3b-msmarco-10k
    model_class=monoT5
    python -m reranking \
        --model_class ${model_class} \
        --reranker_name_or_path ${model_name} \
        --tokenizer_name ${model_name} \
        --corpus ${DATASET_DIR}/RACE/passages \
        --topic ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
        --batch_size 100 \
        --max_length 512 \
        --input_run runs/baseline.${retriever}.race-${split}.passages.run \
        --output runs/reranking.${retriever}+${model_class}.race-${split}.passages.run \
        --top_k 100 \
        --device cuda \
        --fp16
done
done
