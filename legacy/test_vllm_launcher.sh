#!/bin/bash
#SBATCH --account=project_465001640
#SBATCH --output=logs/crux.out       
#SBATCH --error=logs/crux.err
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=120G
#SBATCH --time=0-01:00:00           # Run time (d-hh:mm:ss)

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

source /scratch/project_465001640/personal/dylan/venv/crux_env/bin/activate
export SINGULARITY_BIND="/scratch/project_465001640"

python3 vllm_launcher.py \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dtype bfloat16 --enforce-eager \
    --tensor-parallel-size 8  --max-model-len 8192
