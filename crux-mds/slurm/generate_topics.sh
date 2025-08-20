#!/bin/bash -l
#SBATCH --job-name=crux-mds-topic
#SBATCH --output=logs/crux.out       
#SBATCH --error=logs/crux.err
#SBATCH --partition=dev-g         # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=0-01:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

# Load the environment
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /scratch/project_465001640/personal/dylan/venv/crux_env/bin/activate

export SINGULARITY_BIND="/scratch/project_465001640"
root_dir=/scratch/project_465001640/personal/dylan/datasets/crux

subset=duc04
python3 -m crux.augmentation.gen_topics \
    --config $HOME/crux/configs/default_config.yaml \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --num_gpus 8 \
    --dataset mds --subset $subset \
    --output_dir $root_dir/crux-mds-${subset}/topics \
    --max_new_tokens 128 \
    --max_model_len 8196 \
    --batch_size 32 \
    --load_mode vllm
