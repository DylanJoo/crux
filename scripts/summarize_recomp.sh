#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=recomp
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

# Start the experiment.
for split in testb test;do
# for run_file in runs/baseline.*.race-${split}.passages.run;do
for run_file in runs/reranking.*.race-${split}.passages.run;do
    python3 summarize_ind.py \
        --model_name_or_path fangyuan/nq_abstractive_compressor \
        --model_class seq2seq \
        --template 'Question: \n Document: {P}\n Summary: ' \
        --max_length 512 \
        --batch_size 32 \
        --run_file ${run_file} \
        --topk 30 \
        --topic_file ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
        --passage_dir ${DATASET_DIR}/RACE/passages \
        --output_file outputs/${split}_recomp_psgs.jsonl \
        --truncate 5 \
        --do_irrelevant
done
done

