#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=eval-know
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

# bm25/contriever/splade top10 + vanilla/bartcnndm/recomp
for split in testb;do
for retriever in bm25 contriever splade;do
    aug_method=vanilla
    python3 -m evaluation \
        --judgement_file judgements/race-${split}-${retriever}-top30-${aug_method}.jsonl \
        --dataset_file ${DATASET_DIR}/RACE/shard_data/ratings-gen/8b/metallama3.1-8b-${split}-0.jsonl \
        --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
        --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
        --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --topk 10 \
        --threshold 1 \
        --tag ${retriever}-topk-${aug_method}

    for aug_method in bartsum recomp;do
    for topk in 20;do
        python3 -m evaluation \
            --judgement_file judgements/race-${split}-${retriever}-top30-${aug_method}.jsonl \
            --dataset_file ${DATASET_DIR}/RACE/shard_data/ratings-gen/8b/metallama3.1-8b-${split}-0.jsonl \
            --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
            --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
            --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
            --topk ${topk} \
            --threshold 1 \
            --tag ${retriever}-topk-${aug_method}
    done
    done
done
done
