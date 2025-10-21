"""
This script is largely based on the original script from the Pyserini repository.

see pyserini/search/lucene/_impact_searcher.py
"""
import os
import json
import argparse
from tqdm import tqdm 
from tools import load_topics, batch_iterator

from pyserini.search.lucene import LuceneImpactSearcher

def search(
    index, topics, 
    k, 
    model_name_or_path,
    model_class,
    max_length,
    pooling='mean',
    l2_norm=False,
    batch_size=32,
    query_prefix=None,
    writer=None
):
    # load the query encoder
    searcher = LuceneImpactSearcher(
        index_dir=index,
        query_encoder=model_name_or_path,
        min_idf=0,
    )
    ## monkey patch for pyserini query encoder: https://github.com/castorini/pyserini/blob/master/pyserini/encode/_splade.py#L24
    searcher.query_encoder.device = 'cuda' 
    searcher.query_encoder.model.to('cuda')

    qids = list(topics.keys())
    qtexts = list(topics.values())

    outputs = {}

    for (start, end) in tqdm(
        batch_iterator(range(0, len(qids)), batch_size, True),
        desc='Searching (sparse)',
        total=(len(qids)//batch_size)+1,
    ):
        qids_batch = qids[start: end]
        qtexts_batch = qtexts[start: end]
        hits = searcher.batch_search(
            queries=qtexts_batch, 
            qids=qids_batch,  # inconsistent to dense :(
            threads=10,
            k=k,
        )

        for key, value in hits.items():
            outputs[key] = {h.docid: float(h.score) for h in hits[key]}

            if writer is not None:
                for i in range(len(hits[key])):
                    writer.write(
                        f'{key} Q0 {hits[key][i].docid:4} {i+1} {hits[key][i].score:.5f} faiss\n'
                    )
                writer.close()
    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=100, type=int)
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--topics", default=None, type=str)
    parser.add_argument("--model_name_or_path", default='facebook/contriever-msmarco', type=str)
    parser.add_argument("--model_class", default='contriever', type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--pooling", default='mean', type=str)
    parser.add_argument("--l2_norm", default=False, action='store_true')
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)

    ## load data
    topics = load_topics(args.topics)

    search(
        index=args.index_dir,
        topics=topics,
        k=args.k,
        model_name_or_path=args.model_name_or_path,
        model_class=args.model_class,
        max_length=args.max_length,
        pooling=args.pooling,
        l2_norm=args.l2_norm,
        batch_size=args.batch_size,
        query_prefix=None,
        writer=open(args.output, 'w') if args.output is not None else None,
    )

    print('done')
