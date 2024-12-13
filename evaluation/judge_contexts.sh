#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=ctxjudge
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

split=testb
for aug_method in zs-llmsum;do
    python3 -m evaluation.llm_judge \
        --config configs/mds-decontextualize.llama3-8b.yaml \
        --topic_question_file ${DATASET_DIR}/RACE/ranking/${split}_topics_exam_questions.jsonl \
        --context_file outputs/${split}_${aug_method}_psgs.jsonl \
        --output_file judgements/${split}_${aug_method}_judgements.jsonl  \
        --n_questions 15 \
        --split ${split} \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --model_tag metallama3.1-8b \
        --load_mode vllm \
        --temperature 0.7 \
        --top_p 0.9 \
        --max_new_tokens 5 \
        --ampere_gpu 
done

split=test
for aug_method in zs-llmsum;do
    python3 -m evaluation.llm_judge \
        --config configs/mds-decontextualize.llama3-8b.yaml \
        --topic_question_file ${DATASET_DIR}/RACE/ranking/${split}_topics_exam_questions.jsonl \
        --context_file outputs/${split}_${aug_method}_psgs.jsonl \
        --output_file judgements/${split}_${aug_method}_judgements.jsonl  \
        --n_questions 10 \
        --split ${split} \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --model_tag metallama3.1-8b \
        --load_mode vllm \
        --temperature 0.7 \
        --top_p 0.9 \
        --max_new_tokens 5 \
        --ampere_gpu 
done

