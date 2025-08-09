import asyncio
import requests
from argparse import Namespace
import re
import os
import glob
from collections import defaultdict, OrderedDict
import json
import pickle

from ir_measures import Qrel
from tqdm import tqdm
import pandas as pd

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx : min(ndx + size, l)]


def load_diversity_qrels(path: str) -> list:
    qrels = pd.read_csv(path, sep="\s+", names=["query_id", "iteration", "doc_id", "relevance"])

    diversity_qrels = [Qrel(str(row.query_id), row.doc_id, row.relevance, row.iteration) for row in qrels.itertuples(index=False)]

    return diversity_qrels
    # return ir_measures.read_trec_qrels(path)

def load_query(path, fields=['title', 'problem_statement']):
    queries = {}
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            queries[data.pop('request_id')] = " ".join([data[field] for field in fields])
    return queries

# dylan: try to separate the loading function for NeuCLIR-IR and NeuCLIR-RAG tasks
def load_topic(path):
    """title + description as query for search"""
    topics = {}
    if path.endswith("tsv"):
        with open(path, "r") as f:
            for i, line in enumerate(f):
                qid, qtext = line.split("\t")
                topics[str(qid.strip())] = qtext.strip()

                if (i + 1) == debug:
                    break
    if path.endswith("jsonl"):
        with open(path, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())

                title = data["topics"][0]["topic_title"]
                desc = data["topics"][0]["topic_description"]
                topics[str(data["topic_id"]).strip()] = title + " " + desc
    return topics


def prepreocess(texts):
    pattern = re.compile(r"^(\d+)*\.")
    texts = re.sub(r"\<q\>|\<\/q\>", "\n", texts)
    texts = re.sub(pattern, "\n", texts)
    pattern = re.compile(r"^(\d+)*\.")
    texts = re.sub(pattern, "", texts)
    return texts

def load_subtopics_human(path,
                         args, 
                         raw_topics=None,
                         create_new_subtopics=False):
    # [TODO] AND/OR in the pipeline
    files = [f for f in glob.glob(f"{path}/nuggets_*json")]
    subquestions = {}
    for file in files:
        match = re.search(r"nuggets_(\d+)\.json$", file)
        qid = str(match.group(1))
        data = json.load(open(file, "r"))
        subquestions[qid] = list(data.keys())
        # subanswer[id] = [subanswer for subanswer in subquestions[id]]

    if create_new_subtopics:
        extra_subtopics = generate_complementary_subtopics(args, raw_topics, subquestions)
        for qid in extra_subtopics:
            subquestions[qid].extend(extra_subtopics[qid])

    return subquestions

def load_ratings(path):
    ratings = defaultdict(lambda: defaultdict(lambda: None))
    contexts = defaultdict(lambda: None)
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                id = data['id']
                ratings[id].update({data['pid']: data['rating']})
    return ratings


def load_subtopics(args, topic):
    # TODO: replace with code from the auto-nuggetization team
    subtopics = generate_subtopics(args, topic)
    return subtopics

async def async_load_subtopics(args, topic):
    # [TODO] replace with code from the auto-nuggetization team
    subtopics = await async_generate_subtopics(args, topic)
    return subtopics


def load_corpus_online(path=None):
    class get_content:
        def __len__(self):
            return 1

        def __getitem__(self, doc_id):
            doc = requests.post(url="http://10.162.95.158:5000", json={"collection": "neuclir", "id": doc_id}).json()
            text = doc.get("text", "").replace("\n", " ").strip()
            title = doc.get("title", "").replace("\n", " ").strip()
            return {"text": text, "title": title}

    return get_content()


def load_nuggets(path, include_answer=False):
    if os.path.isdir(path):
        files = [f for f in glob.glob(f"{path}/*")]
    else:
        files = [path]

    nuggets = {}
    for file in files:
        match = re.search(r"nuggets_(\d+)\.json$", file)
        if match:
            qid = str(match.group(1))
            nuggets[qid] = json.load(open(file, "r"))
    return nuggets


def get_judgements_path(args):
    """judgements are computed by running augmentation/gen_ratings.py"""
    judgements_dir = os.path.join(args.crux_dir, args.dataset_name, args.tag)
    path = os.path.join(judgements_dir, f"crux_{args.model}.jsonl")
    return path


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
