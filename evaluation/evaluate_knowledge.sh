#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=eval
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

# [Oracle]
# aug_method=oracle-report
# split=test
# python3 -m evaluation.llm_prejudge \
#     --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --judgement_file judgements/${split}_oracle-report_judgements.jsonl \
#     --threshold 3 \
#     --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#     --rel_threshold 3 \
#     --n_questions 10 \
#     --passage_path outputs \
#     --tag ${aug_method} > ${split}.result
# split=testb
# python3 -m evaluation.llm_prejudge \
#     --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --judgement_file judgements/${split}_oracle-report_judgements.jsonl \
#     --threshold 3 \
#     --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#     --rel_threshold 3 \
#     --n_questions 15 \
#     --passage_path outputs \
#     --tag ${aug_method} > ${split}.result

# aug_method=oracle-passages
# for rel_threshold in 1 3;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --threshold 3 \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --rel_threshold ${rel_threshold} \
#         --n_questions 10 \
#         --passage_path ${DATASET_DIR}/RACE/passages \
#         --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
#         --tag ${aug_method}' (rel='${rel_threshold}')'  >> ${split}.result
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --threshold 3 \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --rel_threshold ${rel_threshold} \
#         --n_questions 15 \
#         --passage_path ${DATASET_DIR}/RACE/passages \
#         --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
#         --tag ${aug_method}' (rel='${rel_threshold}')' >> ${split}.result
# done

# [Vanilla]
# aug_method=vanilla
# for topk in 10 20 30;do
# for retriever in bm25 contriever splade;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 10 \
#         --passage_path ${DATASET_DIR}/RACE/passages \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 15 \
#         --passage_path ${DATASET_DIR}/RACE/passages \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
# done
# done

# [Bartsum]
# aug_method=bartsum
# for topk in 10 20 30;do
# for retriever in bm25 contriever splade;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 10 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 15 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
# done
# done

# [Recomp]
# aug_method=recomp
# for topk in 10 20 30;do
# for retriever in bm25 contriever splade;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 10 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 15 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
# done
# done


### Analysis 1: vanilla + reranking
# aug_method=vanilla+reranking
# for topk in 10 20 30;do
# for retriever in bm25 contriever splade;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/reranking.${retriever}+monoT5.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 10 \
#         --passage_path ${DATASET_DIR}/RACE/passages \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/reranking.${retriever}+monoT5.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 15 \
#         --passage_path ${DATASET_DIR}/RACE/passages \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
# done
# done

### Analysis 2: summary + reranking
# for aug_method in recomp bartsum;do
# for topk in 10 20 30;do
# for retriever in contriever;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/reranking.${retriever}+monoT5.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 10 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}+monoT5-${topk}-${aug_method} >> ${split}.result
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/reranking.${retriever}+monoT5.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 15 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}+monoT5-${topk}-${aug_method} >> ${split}.result
# done
# done
# done

### Analysis 2: vanilla + reranking (only top10)
aug_method=vanilla
topk=10
retriever=contriever
split=test
python3 -m evaluation.llm_prejudge \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
    --threshold 3 \
    --run_file runs/reranking.${retriever}+monoT5.race-${split}.passages.run \
    --topk ${topk} \
    --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
    --n_questions 10 \
    --passage_path ${DATASET_DIR}/RACE/passages \
    --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
split=testb
python3 -m evaluation.llm_prejudge \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
    --threshold 3 \
    --run_file runs/reranking.${retriever}+monoT5.race-${split}.passages.run \
    --topk ${topk} \
    --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
    --n_questions 15 \
    --passage_path ${DATASET_DIR}/RACE/passages \
    --tag ${retriever}-${topk}-${aug_method} >> ${split}.result

# Analysis 2: [Zero-shot-llama-sum]
# aug_method=zs-llmsum
# for topk in 10 20 30;do
# for retriever in bm25 contriever splade;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 10 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 15 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
# done
# done

# aug_method=zs-llmqfsum
# for topk in 10 20 30;do
# for retriever in bm25 contriever splade;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 10 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 15 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.result
# done
# done

## Analysis 3: [Zero-shot-llama-report-gen]
# aug_method=zs-llmrg
# split=test
# python3 -m evaluation.llm_prejudge \
#     --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#     --threshold 3 \
#     --n_questions 10 \
#     --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#     --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#     --tag ${aug_method} >> ${split}.result
# split=testb
# python3 -m evaluation.llm_prejudge \
#     --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#     --threshold 3 \
#     --n_questions 15 \
#     --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#     --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#     --tag ${aug_method} >> ${split}.result
