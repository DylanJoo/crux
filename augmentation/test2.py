import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re
import os
import yaml
import argparse
import json
import numpy as np
from tqdm import tqdm
from glob import glob

from prompts.mds import *

def replace_tags(sent, tag='q'):
    if tag == 'q':
        sent = re.sub(r"\<q\>|\<\/q\>", "\n", sent)
    if tag == 'p':
        sent = re.sub(r"\<p\>|\<\/p\>", "\n", sent)

    pattern = re.compile(r'^\d+\s*[-\\.)]?\s+')
    sent = re.sub(pattern, '', sent)

    pattern = re.compile(r"\n+")
    sent = re.sub(pattern, '\n', sent)
    return sent


def normalize_text(string):
    string = string.strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    return string

def load_passages(path, n=3):
    data = json.load(open(path, 'r'))

    passages = []
    for i, item in enumerate(data['data']):
        example_id = item['example_id']
        if i == 0:
            print(example_id)

        doc_outputs = []
        for doc_output in item['docs']['output']:
            doc_output = normalize_text(doc_output)
            if doc_output == " ":
                doc_outputs.append(["No content."])
            else:
                doc_output = doc_output.strip().split('</p>')[:n]
                doc_output = [replace_tags(o, 'p').strip() for o in doc_output]
                doc_output = [o.strip() for o in doc_output if o.strip() != ""]
                doc_outputs.append(doc_output)

        passages.append({
            "example_id": example_id, 
            "texts": doc_outputs, 
            "docs_full_texts": [normalize_text(d) for d in item["docs"]["full_text"]]
        })
    return passages

logger.info("load passages (and documents)...") 
passages_all = []
for file in tqdm(sorted(glob(os.path.join("/home/dju/datasets/RACE/shard_data/psgs-gen/*-test-*.json")))):
    print(file)
    passages = load_passages(file)
    passages_all += passages

documents_all = {p['example_id']: p['docs_full_texts'] for p in passages_all}
passages_all = {p['example_id']: p['texts'] for p in passages_all}
print(len(passages_all))
