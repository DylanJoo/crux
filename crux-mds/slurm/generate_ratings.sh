#!/bin/bash -l
#SBATCH --job-name=crux-mds-rating
#SBATCH --output=logs/crux.out.%j
#SBATCH --error=logs/crux.err.%j
#SBATCH --partition=small-g         # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --array=0-10%4
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=0-01:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

# Load the environment
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /scratch/project_465001640/personal/dylan/venv/crux_env/bin/activate
export SINGULARITY_BIND="/scratch/project_465001640"

# root_dir=/exp/scale25/artifacts/crux
root_dir=/scratch/project_465001640/personal/dylan/datasets/crux

# subset=duc04
# python3 -m crux.augmentation.gen_ratings \
#     --config $HOME/crux/configs/default_config.yaml \
#     --model meta-llama/Llama-3.3-70B-Instruct \
#     --num_gpus 8 \
#     --dataset mds --subset $subset \
#     --output_dir $root_dir/crux-mds-$subset/judge/v2 \
#     --max_new_tokens 4 \
#     --max_model_len 8196 \
#     --batch_size 32 \
#     --load_mode vllm \
#     --run_path $root_dir/crux-mds-$subset/qrels/qrels.txt \
#     --corpus $root_dir/crux-mds-corpus \

subset=multi_news
python3 -m crux.augmentation.gen_ratings \
    --config $HOME/crux/configs/default_config.yaml \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --num_gpus 4 \
    --dataset mds --subset $subset \
    --output_dir $root_dir/crux-mds-$subset/judge/v2 \
    --max_new_tokens 4 \
    --max_model_len 8196 \
    --batch_size 64 \
    --load_mode vllm \
    --run_path $root_dir/crux-mds-$subset/qrels/qrels.txt \
    --corpus $root_dir/crux-mds-corpus \
    --shard $SLURM_ARRAY_TASK_ID --total_shards 10
