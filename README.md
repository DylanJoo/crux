
### Evaluation data generation

1. Prepare the source dataset: Multi-News and DUC'04
2. Generate the data for the framework: TBD
```
augmentation/generate_xxx.sh
```

3. Compile them into the crux data scheme (default answerability tau is 3)
```
bash create_context_ranking_data.sh
```

4. Get documents and passages
```
bash create_dataset.sh
```

5. Data statisitcs
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

### Main findings
* Table2: The oracle retrieval context for baseline retrievla-augmentation
