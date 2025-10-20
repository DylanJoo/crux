python3 dev.py \
    --run runs/bm25.default.crux-mds-duc04.txt \
    --qrel ./qrel.mds_duc04.txt \
    --filter_by_oracle \
    --judge /home/hltcoe/jhueiju/temp/datasets/crux/crux-mds-duc04/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl

python3 dev.py \
    --run ./qwen3-mds-duc04.run \
    --qrel ./qrel.mds_duc04.txt \
    --filter_by_oracle \
    --judge /home/hltcoe/jhueiju/temp/datasets/crux/crux-mds-duc04/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl

