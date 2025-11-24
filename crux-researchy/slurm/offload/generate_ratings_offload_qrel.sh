module load anaconda3/2024.2
conda activate crux

root_dir=/exp/scale25/artifacts/crux
# root_dir=/home/hltcoe/jhueiju/datasets/crux

# Generate ratings for each researchy questions with top 20
python3 -m crux.augmentation.gen_ratings_offload_qrel \
    --config $HOME/crux/configs/default_config.yaml \
    --dataset researchy \
    --corpus $root_dir/crux-researchy/docs/cw22-b.researchy-v1/corpus.pkl \
    --run_path $root_dir/crux-researchy/runs/run.researchy-all-init-q.bm25.clueweb22-b.txt \
    --output_dir $root_dir/crux-researchy/judge-offload-qrel \
    --shard 0 --total_shards 1 \
    --batch_size 64
