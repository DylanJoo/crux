#!/bin/sh
#SBATCH --job-name=encode-q
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

module load anaconda3/2024.2
conda activate crc

MODEL='Qwen/Qwen3-Embedding-8B'
MULTIJOBS=${HOME}/multigpu.txt

# python -m tevatron.retriever.driver.search \
#     --query_reps /exp/scale25/artifacts/crux/temp/qwen3.crux-mds-duc04.queries.tevatron.pkl \
#     --passage_reps /exp/scale25/artifacts/crux/temp/qwen3.crux-mds-duc04.passages.tevatron.pkl0 \
#     --depth 100 \
#     --batch_size 64 \
#     --save_text \
#     --save_ranking_to /exp/scale25/artifacts/crux/temp/qwen3.crux-mds-duc04.run.txt
#
# python -m tevatron.utils.format.convert_result_to_trec \
#     --input /exp/scale25/artifacts/crux/temp/qwen3.crux-mds-duc04.run.txt \
#     --output /home/hltcoe/jhueiju/crux-scale/qwen3-mds-duc04.run \
#     --remove_query

for mq in +1sq +2sq;do
python -m tevatron.retriever.driver.search \
    --query_reps /exp/scale25/artifacts/crux/temp/qwen3.crux-mds-duc04.queries${mq}.tevatron \
    --passage_reps /exp/scale25/artifacts/crux/temp/qwen3.crux-mds-duc04.passages.tevatron.pkl0 \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to /exp/scale25/artifacts/crux/temp/qwen3.crux-mds-duc04.${mq}.run.txt

python -m tevatron.utils.format.convert_result_to_trec \
    --input /exp/scale25/artifacts/crux/temp/qwen3.crux-mds-duc04.${mq}.run.txt \
    --output /home/hltcoe/jhueiju/crux-scale/qwen3-mds-duc04.${mq}.run \
    --remove_query
done
