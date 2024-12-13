from operator import itemgetter
import os
import datetime
import logging
import argparse 
from tqdm import tqdm
import json

from reranking.crossencoders import monoT5, monoBERT
from reranking.utils import (
    load_topic, 
    load_corpus, 
    load_runs, 
    batch_iterator,
)

def load_reranker(
    model_class, 
    reranker_name_or_path, 
    tokenizer_name=None, 
    device='auto',
    fp16=False
):
    # select backbone model
    model_cls_map = {"monot5": monoT5, "monobert": monoBERT}

    if model_class is not None:
        model_cls = model_cls_map[model_class]
    else:
        for model_cls_key in model_cls_map:
            if model_cls_key.lower() in reranker_name_or_path.lower():
                model_cls = model_cls_map[model_cls_key]
                break
    
    crossencoder = model_cls(
        model_name_or_dir=reranker_name_or_path,
        tokenizer_name=(tokenizer_name or reranker_name_or_path),
        device=device,
        fp16=fp16
    )

    return crossencoder

def rerank(
    topic, 
    corpus, 
    input_run,
    reranker, 
    run_writer,
    top_k,
    batch_size,
    max_length,
    tag
):

    topics = load_topic(topic)
    qids = list(topics.keys())
    qtexts = list(topics.values())
    corpus = load_corpus(corpus)
    runs = load_runs(input_run, topk=top_k, output_score=False)
    qids = [qid for qid in qids if qid in runs]  # only appeared in run

    for qid in tqdm(qids, total=len(qids)):
        result = runs[qid]
        query = topics[qid]
        documents = [corpus[docid] for docid in result]

        # predict
        scores = []
        for batch_docs in batch_iterator(documents, batch_size):
            queries = [query] * len(batch_docs)
            batch_scores = reranker.predict(
                queries=queries,
                documents=[doc['text'] for doc in batch_docs],
                titles=[doc['title'] for doc in batch_docs],
                max_length=max_length
            )
            scores.extend(batch_scores)

        # re-rank the candidate
        hits = {result[idx]: scores[idx] for idx in range(len(scores))}            
        sorted_result = {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)} 

        # write
        for i, (docid, score) in enumerate(sorted_result.items()):
            run_writer.write(f"{qid} Q0 {docid} {str(i+1)} {score} {tag}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=str, default=None)
    parser.add_argument("--reranker_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--input_run", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fp16", default=False, action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)

    run_writer = open(args.output, 'w')

    reranker = load_reranker(
        model_class=args.model_class.lower(),
        reranker_name_or_path=args.reranker_name_or_path,
        tokenizer_name=args.tokenizer_name, 
        device=args.device,
        fp16=args.fp16
    )

    rerank(
        topic=args.topic, 
        corpus=args.corpus,
        input_run=args.input_run,
        reranker=reranker,
        run_writer=run_writer,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_length=args.max_length,
        tag=f'{args.model_class}.rerank'
    )
    run_writer.close()
