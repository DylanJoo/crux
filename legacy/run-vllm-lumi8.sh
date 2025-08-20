#!/bin/bash
#SBATCH --account=project_465001640
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=120G
#SBATCH --time=0-01:00:00           # Run time (d-hh:mm:ss)

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

# Where to store the vLLM server log
VLLM_LOG=${SLURM_JOB_ID}.log

MODEL="meta-llama/Llama-3.3-70B-Instruct"

python -m vllm.entrypoints.openai.api_server --model=$MODEL --tensor-parallel-size 8 --pipeline-parallel-size 1 --gpu-memory-utilization=0.95 --max-model-len 8196 --enforce-eager > $VLLM_LOG &

VLLM_PID=$!

echo "Starting vLLM process $VLLM_PID - logs go to $VLLM_LOG"

# Wait until vLLM is running properly
until curl -sf http://127.0.0.1:8000/health > /dev/null; do
  sleep 2
done

curl localhost:8000/v1/completions -H "Content-Type: application/json" \
     -d "{\"prompt\": \"What would be like a hello world for LLMs?\", \"temperature\": 0, \"max_tokens\": 100, \"model\": \"$MODEL\"}" | json_pp

# To stop job after we have run what we want kill it
kill $VLLM_PID

