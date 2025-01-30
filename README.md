### Prerequisite
```
pip install vllm
pip install transformers
pip install pyserini
pip install ir_measures
```

### CRUX Datasets
The training and testing datasets (test, testb) are available on [Huggingace](#). You can also reproduce the data with the following pipeline.


#### Evaluation data generation pipeline
We have already shared all the dataset (raw, intermediate and final data) in [Huggingface repo](#). But you can also follow the steps to generate and customize.  

1. Multi-document summarization dataset 
- Multi-News and DUC'04: 
We also upload these two raw datasets in our huggingface dataset repo.
> See the original dataset and their terms of use: [Multi-News](https://huggingface.co/datasets/alexfabbri/multi_news) and [DUC'04](https://www-nlpir.nist.gov/projects/duc/data.html)

2. Passages, topics, question generation
You can find the raw generated data in our huggingface dataset repo under [crux/shard_data/](). 
Or you can also reproduce crux with the following scripts. For example, to generate passages. 
```
python3 -m augmentation.gen_passages \
    --config configs/crux-testing-70b.yaml \
    --multi_news_file ${dataset_dir}/multi_news \
    --shard $shard_i --shard_size 1000 \
    --batch_size 2 \
    --tag psgs-gen \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_new_tokens 640 \
    --output_dir ${dataset_dir}/crux/shard_data/
```
While for training pairs, we use smaller (llama-3.1-8B) to generate:
```
dataset=datasets/
python3 -m augmentation.gen_passages \
    --config configs/crux-default-8b.yaml \
    --multi_news_file ${dataset_dir}/multi_news \
    --shard $shard_i --shard_size 1000 \
    --split train \
    --batch_size 2 \ # this could be larger as the model is smaller.
    --tag psgs-gen \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_new_tokens 640 \
    --output_dir ${dataset_dir}/crux/shard_data/
```
Check [scripts](scripts/) for others (topcis & questions).

3. Rating generation (Answerability)
```
python3 -m augmentation.gen_ratings \
    --config configs/crux-testing-70b.yaml \
    --shard $shard_i --shard_size 1000 \
    --batch_size 8 \
    --tag ratings-gen \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_new_tokens 5 \
    --shard_dir ${dataset_dir}/crux/shard_data/
```

4. Compile into ranking task
Compile all the data into IR datasets with passage-level fine-grained relevance labels.
Once we had the ratings, we can create the subsets of `required`, `partially redundant` and `fully redundant`.
```
bash create_ranking_data.sh

# and also the corpus (contains tran/test/testb)
bash create_corpus.sh
```

### Data example
```
# Topic:
# Questions:
# Passages (oracle) and ratings:
```

#### Baselines
TBD


---
#### Others
Data statisitcs
```
## corpus
python -m tools.get_stats --dataset_dir /home/dju/datasets/crux --split testb

## qrels
cat ${DATASET_DIR}/crux/ranking_3/test_qrels_oracle_context_pr.txt  | cut -f 4 -d ' ' | sort | uniq -c 
```

