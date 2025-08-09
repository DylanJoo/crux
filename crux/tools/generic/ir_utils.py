import re
import os
import glob
from collections import defaultdict, OrderedDict
import json
from tqdm import tqdm
import ir_measures
import pandas as pd
import math

def load_run_or_qrel(path, topk=10, threshold=3, threshold_score=-math.inf):
    run_dict = defaultdict(dict)
    with open(path, "r") as f:
        for i, line in enumerate(f):
            try: 
                qid, _, docid, rank, score, _ = line.strip().split()
                if (int(rank) <= topk):
                    run_dict[qid].update({docid: float(score)})
            except:
                qid, _, docid, rel = line.strip().split()
                if int(rel) >= threshold:
                    run_dict[qid].update({docid: float(rel)})
    return run_dict

def load_corpus(path):
    from .text_utils import normalize_doc
    corpus = {}

    if path.endswith('.pkl'):
        import pickle
        with open(path, 'rb') as f:
            corpus = pickle.load(f)
        return corpus

    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            docid = data.get('id', data.get('_id', ''))
            title = data.get('title', "").strip()
            text = data.get('contents', data.get('text', "")).strip()
            text = normalize_doc(text)
            corpus[str(docid)] = {'title': title, 'text': text}
    return corpus

def load_ratings(path):
    ratings = defaultdict(lambda: defaultdict(lambda: None))
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            ratings[data['id']].update({data['docid']: data['rating']})
    return ratings

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

def load_qrels(path, threshold=1):
    data = defaultdict(dict)
    with open(path) as f:
        for line in f:
            qid, _, docid, score = line.strip().split()
            if int(score) >= threshold:
                data[qid].update({docid: int(score)})
    return data

def load_diversity_qrels(path):
    # return pd.read_csv(path, sep='\s+', names=['query_id', 'iteration', 'doc_id', 'relevance'])
    return ir_measures.read_trec_qrels(path)

def load_topics(path, debug=None):
    topics = {}
    if path.endswith('tsv'):
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                qid, qtext = line.split('\t')
                topics[str(qid.strip())] = qtext.strip()
                
                if (i+1) == debug:
                    break
    if path.endswith('jsonl'):
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                topics[data['example_id']] = data['topic'].strip()
                if (i+1) == debug:
                    break
    return topics

def load_reports(path):
    topics = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            topics[data['example_id']] = data['report'].strip()
    return topics

def prepreocess(texts):
    pattern = re.compile(r"^(\d+)*\.")
    texts = re.sub(r"\<q\>|\<\/q\>", "\n", texts)
    texts = re.sub(pattern, '\n', texts)
    pattern = re.compile(r"^(\d+)*\.")
    texts = re.sub(pattern, '', texts)
    return texts     

def load_questions(path):
    questions = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            questions[data.pop('example_id')] = [prepreocess(q) for q in data['questions']]
    return questions

# def load_runs(path, topk=None, output_score=False): # support .trec file only
#     run_dict = defaultdict(list)
#     with open(path, 'r') as f:
#         for line in f:
#             qid, _, docid, rank, score, _ = line.strip().split()
#             if int(rank) <= (9999 or topk):
#                 run_dict[str(qid)] += [(docid, float(rank), float(score))]
#
#     # sort by score and return static dictionary
#     sorted_run_dict = OrderedDict()
#     for qid, docid_ranks in run_dict.items():
#         sorted_docid_ranks = sorted(docid_ranks, key=lambda x: x[1], reverse=False) 
#         if output_score:
#             # sorted_run_dict[qid] = [{docid, rel_score} for docid, rel_rank, rel_score in sorted_docid_ranks]
#             sorted_run_dict[qid] = {docid: rel_score for docid, rel_rank, rel_score in sorted_docid_ranks}
#         else:
#             sorted_run_dict[qid] = [docid for docid, _, _ in sorted_docid_ranks]
#
#     return sorted_run_dict

def sort_and_truncate(run, max_k_dict=None):
    truncated_run = {}
    for qid, docid_scores in run.items():
        topk = max_k_dict[qid]
        sorted_docs = dict(sorted(docid_scores.items(), key=lambda x: x[1], reverse=True)[:topk])
        truncated_run[qid] = sorted_docs
    return truncated_run

def binarize(qrels):
    binarized_qrels = {}
    for qid, docid_scores in qrels.items():
        docid_scores = {docid: 1 for docid, score in docid_scores.items()}
        binarized_qrels[qid] = docid_scores
    return binarized_qrels

