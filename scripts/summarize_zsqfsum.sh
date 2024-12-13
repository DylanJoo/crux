#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=zsqfsum
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

# Start the experiment.
split=testb
rm outputs/${split}_zs-llmqfsum_psgs.jsonl
for run_file in runs/baseline.*.race-${split}.passages.run;do
    python3 summarize_ind.py \
        --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
        --model_class vllm \
        --template "Based on the user's request, summarize the given text.\nUser's request: {Q}\nText: {P}\nSummary: " \
        --template ' on the user request, summarize the following text: {P}\nSummary: ' \
        --batch_size 16 \
        --run_file ${run_file} \
        --topk 100 \
        --topic_file ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
        --passage_dir ${DATASET_DIR}/RACE/passages \
        --output_file outputs/${split}_zs-llmqfsum_psgs.jsonl
done

split=test
rm outputs/${split}_zs-llmqfsum_psgs.jsonl
for run_file in runs/baseline.*.race-${split}.passages.run;do
    python3 summarize_ind.py \
        --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
        --model_class vllm \
        --template "Based on the user's request, summarize the given text.\nUser's request: {Q}\nText: {P}\nSummary: " \
        --batch_size 16 \
        --run_file ${run_file} \
        --topk 30 \
        --topic_file ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
        --passage_dir ${DATASET_DIR}/RACE/passages \
        --output_file outputs/${split}_zs-llmqfsum_psgs.jsonl
done
