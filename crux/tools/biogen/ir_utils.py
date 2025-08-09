import re
import os
from glob import glob
from tqdm import tqdm
import json
from ...tools import batch_iterator, normalize_text
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor

def load_query(path='/exp/scale25/biogen/starter-kit-2025/data/task_b.json', fields=['topic', 'question', 'narrative']):
    queries = {}
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            text = " ".join([item[field] for field in fields])
            queries[item['id']] = text
    return queries

def load_topic(path='/exp/scale25/biogen/topics/task_b.jsonl'):
    topics = {}
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            topics[item.pop('id')] = item
    return topics

def load_subset_corpus(
    glob_path="/exp/scale25/biogen/docs_from_track/jsonl_collection/*.jsonl",
    subset=set(), 
    shard=0, num_shards=1
):
    files = glob(glob_path)
    shard_size = len(files) // num_shards
    files = files[shard * shard_size: (shard + 1) * shard_size]

    def process_file(file_path, docid_set=set()):
        document_list = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                id_field = "URL-hash" if "id" not in data.keys() else "id" 
                content_field = "Clean-Text" if "contents" not in data.keys() else "contents"
                if data[id_field] in docid_set:
                    document_list.append({'id': data[id_field], 'contents': data[content_field]})
        return document_list

    corpus = {}
    with ThreadPoolExecutor(32) as executor:
        for batch_files in tqdm(
            batch_iterator(files, 100), desc="Processing files", total=len(files) // 100+1
        ): 
            found_document_list = sum(
                    executor.map(
                        lambda x: process_file(x[0], x[1]), 
                        [(file_path, subset) for file_path in batch_files]),
                    []
            )

            for doc in found_document_list:
                corpus[doc['id']] = doc
                subset.discard(doc['id'])

    print(f"Total documents found: {len(corpus)}. Size of the subset documents: {len(subset)}")
    return corpus
