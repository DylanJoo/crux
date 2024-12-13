#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=zsrg
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
rm outputs/${split}_zs-llmrg_psgs.jsonl
python3 reportgen.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --model_class vllm \
    --template 'Instruction: {Q}\nWrite a 300 words report as response.\n\nResponse: ' \
    --batch_size 16 \
    --topic_file ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
    --passage_dir ${DATASET_DIR}/RACE/passages \
    --output_file outputs/${split}_zs-llmrg_psgs.jsonl

split=test
rm outputs/${split}_zs-llmrg_psgs.jsonl
python3 reportgen.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --model_class vllm \
    --template 'Instruction: {Q}\nWrite a 300 words report as response.\n\nResponse: ' \
    --batch_size 16 \
    --topic_file ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
    --passage_dir ${DATASET_DIR}/RACE/passages \
    --output_file outputs/${split}_zs-llmrg_psgs.jsonl
