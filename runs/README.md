# Evaluation Results

We evaluate a few first-stage retrieval methods:
- BM25
- Contriever (`facebook/contriever-msmarco`)
- SPLADE v3 (`naver/splade-v3`)
- Qwen3-embedding-8b (`Qwen/Qwen3-Embedding-8B`)

> The evaluation script, please refer to [https://github.com/DylanJoo/crux/tree/module?tab=readme-ov-file#evaluation](https://github.com/DylanJoo/crux/tree/module?tab=readme-ov-file#evaluation)

## CRUX-MDS-DUC04
| method                                    | Metric  | Score  | Metric  | Score  | Metric       | Score  | Metric | Score  |
|-------------------------------------------|---------|--------|---------|--------|--------------|--------|--------|--------|
| bm25.default.crux-mds-duc04.txt           | P@10 | 0.5140 | nDCG@10 | 0.5298 | alpha_nDCG@10 | 0.4454 | Cov@10 | 0.5444 | 
| contriever-ft.crux-mds-duc04.txt          | P@10 | 0.7020 | nDCG@10 | 0.7119 | alpha_nDCG@10 | 0.5515 | Cov@10 | 0.6201 | 
| splade-v3.crux-mds-duc04.txt              | P@10 | 0.6800 | nDCG@10 | 0.7035 | alpha_nDCG@10 | 0.5579 | Cov@10 | 0.6241 | 
| qwen3-embedding-8b.crux-mds-duc04.txt     | P@10 | 0.7380 | nDCG@10 | 0.7583 | alpha_nDCG@10 | 0.6077 | Cov@10 | 0.6637 | 

## CRUX-MDS-Multi_news (small)
| method                                       | Metric | Score  | Metric  | Score  | Metric       | Score  | Metric | Score  |
|----------------------------------------------|--------|--------|---------|--------|--------------|--------|--------|--------|
 bm25.default.crux-mds-multi_news.txt          | P@10 | 0.2610 | nDCG@10 | 0.4122 | alpha_nDCG@10 | 0.4415 | Cov@10 | 0.4622 | 
 contriever-ft.crux-mds-multi_news.txt         | P@10 | 0.3380 | nDCG@10 | 0.4801 | alpha_nDCG@10 | 0.4934 | Cov@10 | 0.5364 | 
 splade-v3.crux-mds-multi_news.txt             | P@10 | 0.3420 | nDCG@10 | 0.5071 | alpha_nDCG@10 | 0.5166 | Cov@10 | 0.5355 | 
 qwen3-embedding-8b.crux-mds-multi_news.txt    | P@10 | 0.3740 | nDCG@10 | 0.5530 | alpha_nDCG@10 | 0.5966 | Cov@10 | 0.6087 | 

## NeuCLIR (with CRUX evaluation)
| method                                    | Metric  | Score  | Metric  | Score  | Metric       | Score  | Metric | Score  |
|-------------------------------------------|---------|--------|---------|--------|--------------|--------|--------|--------|
