#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=oracle
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

## 1. Extract the oracle context (report)
split=test
python3 oracle.py \
    --multi_news_file ${DATASET_DIR}/multi_news \
    --split ${split} \
    --output_file outputs/${split}_oracle-report_psgs.jsonl \
    --tag report 

split=testb
python3 oracle.py \
    --duc04_file ${DATASET_DIR}/duc04 \
    --split ${split} \
    --output_file outputs/${split}_oracle-report_psgs.jsonl \
    --tag report 

