#!/bin/bash -l
#SBATCH --job-name=crux-mds-subquestions
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

subset=multi_news
python3 -m crux.augmentation.gen_subquestions \
    --config $HOME/crux-scale/configs/default_config.yaml \
    --dataset mds --subset ${subset} \
    --output_dir $root_dir/crux-mds-${subset}/subtopics \
    --max_new_tokens 512 \
    --batch_size 32 \
    --n_subquestions 10 \
    --load_mode vllm
