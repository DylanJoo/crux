import logging
logger = logging.getLogger(__name__)

import re
import os
import json
import argparse
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from chatnoir_api.v1 import search
from chatnoir_api import cache_contents

from ...tools import batch_iterator
from ...tools.researchy.ir_utils import sort_and_truncate, load_query, load_runs
import hashlib
from urllib.parse import urlparse

def get_content(uuid=None, trec_id=None):
    try:
        raw_html = cache_contents(uuid or trec_id, index='clueweb22/b', plain=True)
        soup = BeautifulSoup(raw_html, "html.parser")
        soup = BeautifulSoup(soup.get_text(separator=" ", strip=True), "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text
    except:
        logger.warning(f"Failed to retrieve content for {uuid or trec_id}")
        return ""

def CluewebURLHash(url): # this is the function used in the researchy question (probably)
    parsed_url = urlparse(url)
    clean_url = parsed_url.netloc + parsed_url.path
    md5_hash = hashlib.md5(clean_url.encode()).hexdigest()
    return md5_hash.upper()

def chatnoir_search(qid, x, k):
    try:
        results = search(x, index='clueweb22/b')[:k]
        doc_urls = [r.target_uri for r in results]
        docids = [CluewebURLHash(url) for url in doc_urls]
        hits = {qid: {docid: float(r.score) for docid, r in zip(docids, results)}}

        documents = {r.trec_id: {
            "CluewebURLHash": docid,
            "target_uri": r.target_uri, 
            "title": str(r.title),
        } for r, docid in zip(results, docids)}

        return hits, documents
    except:
        logger.warning(f"Failed to search for query {qid}: {x}")
        return qid, {}

def get_clueweb_run(topics, batch_size, k, api_key, output_dir=None):
    qids = list(topics.keys())
    qtexts = list(topics.values())

    # search
    qids_failed, docids_failed = [], []
    corpus = {}
    with ThreadPoolExecutor() as executor:
        for (start, end) in tqdm(
            batch_iterator(range(0, len(qids)), batch_size, True),
            desc='Searching (ChatNoir)', 
            total=(len(qids)//batch_size)+1,
        ):
            runs = {}
            qids_batch = qids[start: end]
            qtexts_batch = qtexts[start: end]

            batch = list(
                executor.map(
                    lambda x: chatnoir_search(x[0], x[1], k),
                    list(zip(qids_batch, qtexts_batch))
                )
            )
            for hits, documents in batch:
                if isinstance(hits, str): # empty hits, failed qid returned
                    qids_failed.append(hits)
                else:
                    runs.update(hits)
                    corpus.update(documents)

            # sorted
            sorted_runs = sort_and_truncate(runs, k)

            # write
            with open(os.path.join(output_dir, 'chatnoir-researchy.run'), 'a') as f:
                for qid in sorted_runs:
                    for rank, (docid, score) in enumerate(sorted_runs[qid].items(), start=1):
                        f.write(f'{qid} Q0 {docid} {rank} {score:.5f} {trec_id}\n')


def get_clueweb_corpus(corpus, output_dir=None):
    with ThreadPoolExecutor() as executor:
        for batch_docids in tqdm(
            batch_iterator(list(corpus.keys())), 
            desc='Get documents', 
            total=(len(corpus)//batch_size)+1
        ):
            batch_contents = list(
                executor.map(
                    lambda x: get_content(uuid=None, trec_id=x), 
                    batch_docids
                )
            )
            for docid, doc_text in zip(batch_docids, batch_contents):
                if doc_text is None:
                    corpus[docid]['contents'] = "failed to retrieve."
                else:
                    corpus[docid]['contents'] = doc_text

    # write
    # if writer is not None:
    with open(os.path.join(output_dir, f'corpus-researchy-clueweb22-b-v0.jsonl'), 'w') as f:
        for docid in corpus:
            f.write(json.dumps({
                'CluewebURLHash': corpus[docid]['CluewebURLHash'],
                'trec_id': docid,
                'target_uri': corpus[docid]['target_uri'],
                'title': corpus[docid]['title'],
                'contents': corpus[docid]['contents'],
            }, ensure_ascii=False) + '\n')

    # save failed queries and documents
    with open(os.path.join(output_dir, 'failed_queries.txt'), 'w') as f:
        for qid in qids_failed:
            f.write(f"{qid}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--api-key", default="pXnxD5KUN8gjZumDQwnoMwxOz0tmrML3", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--debug", default=False, action='store_true', help="Enable debug mode")
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    topics = load_query(debug=args.debug)

    # Runs
    if os.path.exists(os.path.join(args.output_dir, 'chatnoir-researchy.run')):
        runs = load_runs(os.path.join(args.output_dir, 'chatnoir-researchy.run'))
        topics = {qid: topics[qid] for qid in topics if (qid not in runs)}
    else:
        open(os.path.join(args.output_dir, 'chatnoir-researchy.run'), 'w').close()

    get_clueweb_run(
        topics=topics,
        batch_size=args.batch_size, 
        k=args.k,
        api_key=args.api_key,
        output_dir=args.output_dir,
    )
    print('done')

    # Corpus
    # if os.path.exists(os.path.join(args.output_dir, 'corpus-researchy-clueweb22-b-v0.jsonl')):
    #     corpus_existed = load_corpus(os.path.join(args.output_dir, 'corpus-researchy-clueweb22-b-v0.jsonl'))
    #     runs = load_runs(os.path.join(args.output_dir, 'chatnoir-researchy.run'))
    # else:
    #     open(os.path.join(args.output_dir, 'corpus-researchy-clueweb22-b-v0.jsonl'), 'w').close()
