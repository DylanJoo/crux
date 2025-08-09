import os
import glob
import collections 
import json
from tqdm import tqdm

def load_searcher(path, dense=False):
    if dense:
        from pyserini.search.faiss import FaissSearcher
        searcher = FaissSearcher(path, None)
    else:
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(path)
        searcher.set_bm25(k1=0.9, b=0.4)
    return searcher

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def load_topics(path):
    topics = {}
    with open(path, 'r') as f:
        for line in f:
            qid, qtext = line.split('\t')
            topics[str(qid.strip())] = qtext.strip()
    return topics

def load_corpus(path, allow_missing=False):
    if allow_missing:
        empty = {'title': "", 'text': ""}
        corpus = collections.defaultdict(lambda: empty)
    else:
        corpus = {}

    id_field = 'id'
    content_field = 'contents'

    if os.path.isdir(path):
        files = [f for f in glob.glob(f'{path}/*jsonl')]
    else:
        files = [path]

    for file in tqdm(files, total=len(files)):
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())

                if "id" not in data.keys():
                    id_field = "_id"

                if "contents" not in data.keys():
                    content_field = "text"

                docid = data[id_field] 
                title = data.get('title', "").strip()
                text = data.get(content_field, "").strip()
                corpus[str(docid)] = {'title': title, 'text': text}
    return corpus

def load_runs(path, topk=10, output_score=False): # support .trec file only
    run_dict = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if int(rank) <= topk:
                run_dict[str(qid)] += [(docid, float(rank), float(score))]

    # sort by score and return static dictionary
    sorted_run_dict = collections.OrderedDict()
    for qid, docid_ranks in run_dict.items():
        sorted_docid_ranks = sorted(docid_ranks, key=lambda x: x[1], reverse=False) 
        if output_score:
            sorted_run_dict[qid] = [(docid, rel_score) for docid, rel_rank, rel_score in sorted_docid_ranks]
        else:
            sorted_run_dict[qid] = [docid for docid, _, _ in sorted_docid_ranks]

    return sorted_run_dict
