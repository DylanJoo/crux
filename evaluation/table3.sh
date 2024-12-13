#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=eval
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

rm -rf test.table3
rm -rf testb.table3

TAU=3
W=0.5

# [vanilla]
aug_method=vanilla
topk=10
for split in test testb;do
for retriever in bm25 contriever splade;do
python3 -m evaluation.llm_prejudge \
    --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
    --dataset_dir ${DATASET_DIR}/crux \
    --rel_subset 3 \
    --split ${split} \
    --threshold ${TAU} \
    --weighted_factor $W \
    --passage_path ${DATASET_DIR}/crux/passages \
    --judgement_file ${DATASET_DIR}/crux/ranking_${TAU}/${split}_judgements.jsonl \
    --run_file runs/baseline.${retriever}.race-${split}.passages.run \
    --topk ${topk} \
    --tag ${retriever}-${topk}-${aug_method} >> ${split}.table3
done
done

# [vanilla] + [reranking]
for split in test testb;do
for retriever in bm25 contriever splade;do
python3 -m evaluation.llm_prejudge \
    --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
    --dataset_dir ${DATASET_DIR}/crux \
    --rel_subset 3 \
    --split ${split} \
    --threshold ${TAU} \
    --weighted_factor $W \
    --passage_path ${DATASET_DIR}/crux/passages \
    --judgement_file ${DATASET_DIR}/crux/ranking_${TAU}/${split}_judgements.jsonl \
    --run_file runs/reranking.${retriever}+monoT5.race-${split}.passages.run \
    --topk ${topk} \
    --tag ${retriever}-reranking-${topk}-${aug_method} >> ${split}.table3
done
done
