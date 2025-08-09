#!/bin/sh
#SBATCH --job-name=6hr.crux-eval
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate rag

# root
cd /home/dju/rag-rerank/src 
data=test

judge_model=meta-llama/Llama-3.1-70B-Instruct
max_length=-1
mkdir -p logs/$judgement_model

# Oracle summary
result_file=results/$max_length/${data}-oracle_k.jsonl
file_name=${result_file##*/}
output_file=logs/$judge_model/oracle/${data}-oracle-report.log
mkdir -p ${output_file%/*}

echo "Evaluating oracle sumamry: " $file_name
python3 -m evaluation.llmjudge.retrieval_augmentation_generation \
    --topic_file /home/dju/datasets/crux/ranking_5/${data}_topics.jsonl \
    --corpus_dir /home/dju/datasets/crux/passages/ \
    --qrels_file /home/dju/datasets/crux/ranking_3/${data}_qrels_pr.txt \
    --judgement_file /home/dju/datasets/crux/ranking_3/${data}_oracle-passages_judgements.jsonl \
    --model_name_or_path $judge_model \
    --threshold 3 \
    --gamma 0.5  \
    --num_gpus 4 \
    --used_field report \
    --result_jsonl $result_file > $output_file

# Oracel retrieval
for result_file in results/oracle/${data}-oracle*n${max_length}*;do
    file_name=${result_file##*/}
    output_file=logs/$judge_model/oracle/${file_name}
    echo "Evaluating oracle rag report: " $file_name

    python3 -m evaluation.llmjudge.retrieval_augmentation_generation \
        --topic_file /home/dju/datasets/crux/ranking_5/${data}_topics.jsonl \
        --corpus_dir /home/dju/datasets/crux/passages/ \
        --qrels_file /home/dju/datasets/crux/ranking_3/${data}_qrels_pr.txt \
        --judgement_file /home/dju/datasets/crux/ranking_3/${data}_oracle-passages_judgements.jsonl \
        --model_name_or_path $judge_model \
        --threshold 3 \
        --gamma 0.5  \
        --num_gpus 4 \
        --result_jsonl $result_file > $output_file
done
