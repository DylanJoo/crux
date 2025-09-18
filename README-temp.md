# Controlled Retrieval-augmented Context Evaluation for Long-form RAG

### Installation
- Install crux from source
```shell
git clone https://github.com/DylanJoo/crux
cd crux
uv pip install -e .
```
- Prerequisite (recommend to use container + python venv)
TBD

### Available Evaluation Data
1. CRUX-MDS (Multi-document summarization)
The CRUX evaluation datasets were first built upon multi-document summarization as their controllability. We use Multi-News and DUC04 for creating the evaluation dataset.

2. CRUX-TREC
In addition to MDS, we add TREC-style report generation task: NeuCLIR'24 and BioGen'24 as another two evaluation datsets.

> Note that crux-mds is purely synthetic dataset only with our human verficiation. The crux-neuclir and trec-biogen are mostly done by human annotators (See details below)

### Components
There are a few required components.

* Answerability
Here is an example command to obtain sub-question scores:
```shell
python -m crux.augmentation.gen_ratings \
    --shard_dir <crux_dir> --shard_size 1000 \
    --config configs/scale.llama3-70b.yaml \
    --split train \
    --model llama3.3-70b-instruct \
    --model_tag metallama3.3-70b \
    --port 4000 \
    --tag ratings-gen \
    --load_mode api \
    --temperature 0 --top_p 1 \
    --max_new_tokens 3 \
    --output_dir <crux_dir> \
    --ampere_gpu
```
    
```
python3 -m augmentation.gen_ratings \
    --shard_dir <crux_dir> --shard_size 1000 \
    --config configs/scale.llama3-70b.yaml \
    --split test \
    --model llama3.3-70b-instruct \
    --model_tag metallama3.3-70b \
	--port 4000 \
    --tag ratings-gen/70b \
    --load_mode api \
	--temperature 0 --top_p 1 \
    --max_new_tokens 3 \
	--output_dir <crux_dir> \
    --ampere_gpu
```

# Other Instructions

### Prepare source multi-document summarization dataset
- Multi-News and DUC'04: 
We also upload these two raw datasets in our huggingface dataset repo. See the original dataset and their terms of use: [Multi-News](https://huggingface.co/datasets/alexfabbri/multi_news) and [DUC'04](https://www-nlpir.nist.gov/projects/duc/data.html).

### Evaluation data generation
We have already shared all the dataset (raw, intermediate and final data) in [Huggingface repo](#). But you can also follow the steps to generate and customize. 

## Passages, topics, questions 
Here is an example of generating passages. And you will find the generated data in 
`<path-to-crux>/shard_data/psg-gen/metallama3.1-8b-{train/test/testb}-${shard_i}.json`
    * Passages (mulit-news train and test)
	* Topics and Questions (multi-news test and DUC'04)
	* To run them in parallel, you can set the `shard` parameter and assign the `shard size (size of one shard)`

```
python3 -m augmentation.gen_passages  \
	--shard ${shard_i} --shard_size 1000 \
	--multi_news_file <path-to-multi_news> \
	--config configs/mds-decontextualize.llama3-8b.yaml \
	--split train \
	--model meta-llama/Meta-Llama-3.1-8B-Instruct \
	--model_tag metallama3.1-8b \
	--tag psgs-gen \
	--load_mode vllm \
	--temperature 0.7 \
	--max_new_tokens 640 \
	--output_dir <path-to-crux>/shard_data/
	--ampere_gpu
done
```
The topic/question generation can be found in `script/`

### Answerability

```
for split in train test;do
python3 -m augmentation.gen_ratings \
	--shard_dir <path-to-crux>/shard_data --shard_size 1000 \
	--config configs/mds-decontextualize.llama3-8b.yaml \
	--split ${split} \
	--model meta-llama/Meta-Llama-3.1-8B-Instruct \
	--model_tag metallama3.1-8b \
	--tag ratings-gen \
	--load_mode vllm \
	--temperature 0 --top_p 1 \
	--max_new_tokens 3 \
	--output_dir <path-to-crux>/shard_data \
	--ampere_gpu
done

python3 -m augmentation.gen_ratings \
    --shard_dir <path-to-crux>/shard_data --shard_size 1000 \
    --config configs/mds-decontextualize.llama3-8b.yaml \
    --split testb \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --model_tag metallama3.1-8b \
    --tag ratings-gen/8b \
    --load_mode vllm \
	--temperature 0 --top_p 1 \
    --max_new_tokens 3 \
	--output_dir <path-to-crux>/shard_data \
    --ampere_gpu
```

### Data example
```
# Topic:
# Questions:
# Passages (oracle) and ratings:
```

- Compile them into the crux data scheme (default answerability tau is 3)
Once we had the ratings, we can create the subsets of `required`, `partially redundant` and `fully redundant`.

```
bash create_context_ranking_data.sh
```

- Get documents and passages
```
bash create_dataset.sh
```

- Data statisitcs
```
## context
python -m tools.get_stats --dataset_dir /home/dju/datasets/crux --split testb

## ranking qrels
cat ${DATASET_DIR}/crux/ranking_3/test_qrels_oracle_context_pr.txt  | cut -f 4 -d ' ' | sort | uniq -c 
```

### Baseline settings
* The first-stage retrieval: BM25, contriever-MS, SPLADE-MS
```
Indexing and Retrieve top-100
```

* The former-stage augmentation: vanilla, BART-summarization, ReComp summarization

### Example
```json
{
  "id": "example_id",
  "topic": "What is the impact of climate change on polar bears?",
  "questions": [
    "How does climate change affect polar bear habitats?",
    "What are the main threats to polar bears due to climate change?"
    ... (more questions)
  ],
  "passages": [
    {
      "contents": "Climate change is causing the Arctic ice to melt, which is crucial for polar bears.",
      "rating": 5
    },
    {
      "text": "Rising temperatures are leading to habitat loss for polar bears.",
      "rating": 4
    }
  ]
}
```
 

