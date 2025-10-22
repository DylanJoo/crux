import os
import json
import argparse
from tqdm import tqdm 
from ...tools import batch_iterator
from pyserini.search.lucene import LuceneSearcher

def search(index, k1, b, topics, batch_size, k, writer=None, stemming=True):

    searcher = LuceneSearcher(index)
    searcher.set_bm25(k1=k1, b=b)

    if stemming is False:
        from pyserini.analysis import Analyzer, get_lucene_analyzer
        analyzer = get_lucene_analyzer(stemming=False)
        print(f'Using no stemming analyzer: {analyzer}')
        searcher.set_analyzer(analyzer)

    qids = list(topics.keys())
    qtexts = list(topics.values())

    outputs = {}

    for (start, end) in tqdm(
        batch_iterator(range(0, len(qids)), batch_size, True),
        desc='Searching (bm25)', 
        total=(len(qids)//batch_size)+1,
    ):
        qids_batch = qids[start: end]
        qtexts_batch = qtexts[start: end]
        batch_hits = searcher.batch_search(
            queries=qtexts_batch, 
            qids=qids_batch, 
            threads=32,
            k=k,
        )

        for qid, hits in batch_hits.items():
            outputs[qid] = {h.docid: float(h.score) for h in hits}

            if writer is not None:
                for i in range(len(hits)):
                    writer.write(f'{qid} Q0 {hits[i].docid} {i+1} {hits[i].score:.5f} pyserini-k1_{k1},b_{b}\n')

    # close it
    if writer is not None:
        writer.close()
    return outputs  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1",type=float, default=0.9) # 0.5 # 0.82
    parser.add_argument("--b", type=float, default=0.4) # 0.3 # 0.68
    parser.add_argument("--k", default=100, type=int)
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--query", default=None, type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)

    ## load data # [todo] 
    from retrieve.utils import load_query
    topics = load_query(args.query)

    search(
        index=args.index, 
        k1=args.k1, 
        b=args.b, 
        topics=topics,
        batch_size=args.batch_size, 
        k=args.k,
        output=output,
        writer=open(args.output, 'w') if args.output is not None else None,
    )

    print('done')
