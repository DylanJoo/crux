import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import json
import re
import os
import string
import time
from glob import glob
from collections import defaultdict

def update_tokenizer(tokenizer, max_n_contexts=10):
    tokenizer.add_special_tokens({"additional_special_tokens": ["<cls>"]})
    return tokenizer

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def load_topics(path):
    topics = {}

    with open(path) as f:
        for line in f:
            id, text = line.split('\t')
            topics[id] = text.strip()
    return topics

def load_passages(path):
    passages = {}
    if os.path.exists(path) is False:
        return passages

    if os.path.isdir(path):
        paths = glob(os.path.join(path, f"*.jsonl"))
    else:
        paths = [path]

    for file in paths:
        with open(file) as f:
            for line in f:
                item = json.loads(line.strip())
                example_id = item['id']
                passages[example_id] = item['contents']
    return passages

def load_judgements(path):
    judgements = defaultdict(lambda: defaultdict(lambda: None))
    contexts = defaultdict(lambda: None)
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                example_id = data['example_id']
                judgements[example_id].update({data['pid']: data['rating']})
                if 'contents' in data.keys():
                    contexts[example_id].update({data['pid']: data['contents']})
    return judgements, contexts

def load_run(path, topk=9999):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            item = line.strip().split()
            if int(item[3]) <= topk:
                data[item[0]].append( (item[2], float(item[4])) )

    for id in data:
        data[id] = [docid for docid, _ in sorted(data[id], key=lambda x: x[1])]
    return data

def load_model(model_name_or_path, model_class='causuallm', dtype=None, load_mode=None):

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    from .models import FiDT5
    MODEL_CLASS = {"fid": FiDT5, "seq2seq": AutoModelForSeq2SeqLM, "causallm": None}[model_class]

    logger.info(f"Loading {model_name_or_path} ({model_class}) in {dtype}...")
    logger.warn(f"Use generator.{load_mode}")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    if model_class == 'causallm': # one model may larger than a gpu
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            load_in_8bit=True,
            torch_dtype=(dtype or torch.float16)
        ).to('cuda')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    else:
        model = MODEL_CLASS.from_pretrained(model_name_or_path).to('cuda')

    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    return model, tokenizer
