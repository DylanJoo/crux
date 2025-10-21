"""
This script is largely based on the original script from the Pyserini repository.
Check more query encoder specific options in the original script, 
see pyserini/search/faiss/__main__.py 
"""
import os
import json
import argparse
from tqdm import tqdm 
from tools import load_topics, batch_iterator

from pyserini.search.faiss import FaissSearcher
from pyserini.search.faiss.__main__ import init_query_encoder

# non-async search
def search(query, searcher, service_name, limit=100, **kwargs):

    if ('lsr' in retriever) and args.overwrite:
        from pyserini.search.lucene import LuceneImpactSearcher
        searcher = LuceneImpactSearcher(
            index_dir='/exp/scale25/artifacts/crux/temp/splade-v3.crux.passages.lucene',
            query_encoder="naver/splade-v3",
            min_idf=0,
        )
        searcher.query_encoder.device = 'cuda' 
        searcher.query_encoder.model.to('cuda')
        writer = open(f"{retriever}.run", 'w')

def search(
    index_dir, 
    topic_path, 
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

    if 'bm25' in index_dir:
        type = 'lexical'
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(index_dir)
        searcher.set_bm25(k1=0.9, b=0.4)
        writer = open(f"{retriever}.run", 'w')

    elif 'splade' in index_dir:
        type = 'learned sparse'
        searcher = 0
    else:
        type = 'dense'
        query_encoder = init_query_encoder(
            encoder_class=model_class,
            encoder=model_name_or_path,
            tokenizer_name=model_name_or_path,
            topics_name=None, # we input outselves
            encoded_queries=None,
            device='cuda',
            max_length=max_length,
            pooling=pooling,
            l2_norm=l2_norm,
            prefix=query_prefix,
        )
        searcher = FaissSearcher(index, query_encoder)

    outputs = {}
    for (start, end) in tqdm(
        batch_iterator(range(0, len(topic)), batch_size, True),
        desc=f'Searching ({type})',
        total=(len(qids)//batch_size)+1,
    ):
        qids_batch = qids[start: end]
        qtexts_batch = qtexts[start: end]
        hits = searcher.batch_search(
            queries=qtexts_batch, 
            q_ids=qids_batch, 
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
