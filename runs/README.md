# Evaluation Results

We adpot the following first-stage retrieval methods for evaluation:
- BM25
- Contriever (`facebook/contriever-msmarco`)
- SPLADE v3 (`naver/splade-v3`)
- Qwen3-embedding-8b (`Qwen/Qwen3-Embedding-8B`)


## CRUX-MDS-DUC04
| method                                    | Metric  | Score  | Metric  | Score  | Metric       | Score  | Metric | Score  |
|-------------------------------------------|---------|--------|---------|--------|--------------|--------|--------|--------|
| bm25.default.crux-mds-duc04.txt           | P@10 | 0.5140 | nDCG@10 | 0.5298 | alpha_nDCG@10 | 0.4454 | Cov@10 | 0.5444 | 
| contriever-ft.default.crux-mds-duc04.txt  | P@10 | 0.7020 | nDCG@10 | 0.7119 | alpha_nDCG@10 | 0.5515 | Cov@10 | 0.6201 | 
| splade-v3.default.crux-mds-duc04.txt      | P@10 | 0.6800 | nDCG@10 | 0.7035 | alpha_nDCG@10 | 0.5579 | Cov@10 | 0.6241 | 
| qwen3-embedding-8b.crux-mds-duc04.txt     | P@10 | 0.7380 | nDCG@10 | 0.7583 | alpha_nDCG@10 | 0.6077 | Cov@10 | 0.6637 | 

## CRUX-MDS-DUC04
| method                                    | Metric  | Score  | Metric  | Score  | Metric       | Score  | Metric | Score  |
|-------------------------------------------|---------|--------|---------|--------|--------------|--------|--------|--------|

## NeuCLIR (with CRUX evaluation)
| method                                    | Metric  | Score  | Metric  | Score  | Metric       | Score  | Metric | Score  |
|-------------------------------------------|---------|--------|---------|--------|--------------|--------|--------|--------|

