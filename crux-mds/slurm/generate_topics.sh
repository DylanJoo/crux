#!/bin/bash -l
#SBATCH --job-name=crux-mds         
#SBATCH --output=logs/crux.out       
#SBATCH --error=logs/crux.err
#SBATCH --partition=dev-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --time=0-00:05:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001339 # Project for billing

# Load the environment
module use /appl/local/csc/modulefiles/
module load pytorch/2.7
source /scratch/project_465001640/personal/dylan/venv/crux_env/bin/activate

export SINGULARITY_BIND="/scratch/project_465001339,/scratch/project_465001640"
export NUMEXPR_MAX_THREADS=4

# Set the path to the multijobs file
root_dir=/scratch/project_465001339/crux

for subset in multi_news duc04;do
python3 -m crux.augmentation.gen_topics \
    --config $HOME/crux/configs/default_config.yaml \
    --dataset mds --subset $subset \
    --output_dir $root_dir/crux-mds-${subset}/topics \
    --max_new_tokens 128 \
    --batch_size 32 \
    --load_mode vllm
    #SBATCH --output=logs/crux-mds.%j.out       
    #SBATCH --error=logs/crux-mds.%j.err
done
