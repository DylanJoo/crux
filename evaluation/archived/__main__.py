import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import re
import argparse
import json
import numpy as np
from copy import copy
import random
from collections import defaultdict
from tqdm import tqdm
from glob import glob

from transformers import AutoTokenizer

def load_qrel(path, threshold=0):
    data = defaultdict(list)
    if path is None:
        return None
    with open(path) as f:
        for line in f:
            item = line.strip().split()
            if int(item[3]) >= threshold:
                data[item[0]].append( item[2] )
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--judgement_file", type=str, default=None)
    parser.add_argument("--dataset_file", type=str, default=None, help="File path to the file with generated texts.")
    parser.add_argument("--topics", type=str, default=None, help="File path to the topics.")
    parser.add_argument("--qrels", type=str, default=None, help="File path to the qrels.")
    parser.add_argument("--generator_name", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=3)
    parser.add_argument("--rel_threshold", type=float, default=1)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.generator_name)

    # load qrels 
    qrels = load_qrel(args.qrels, threshold=args.rel_threshold)
    topics = {}
    with open(args.topics) as f:
        for line in f:
            id, text = line.split('\t')
            topics[id] = text.strip()

    # load outputs 
    judgements_all = {}
    contexts_all = {}
    judge_LLM = "metallama3.1-8b"
    if 'passages' not in args.tag:
        logger.info("load judgements ...") 
        with open(args.judgement_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())

                try:
                    judgements_all[data['example_id']] = data['judgement_all'][:args.topk]
                except:
                    judgements_all[data['example_id']] = data['judgements'][:args.topk]
                contexts_all[data['example_id']] = data['contexts'][:args.topk]
                judge_LLM = data['judge_LLM']

    # load graded passages
    outputs = {'coverage': [], 'density': [], 'num_segs': [], 'num_tokens': []}
    with open(args.dataset_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())

            example_id = data['example_id']
            questions = data['questions']
            ratings = np.array(data['ratings'])
            passages = []
            for i, plist in enumerate(data['passages']):
                for passage in plist:
                    passages.append(passage)
            
            if example_id not in topics:
                continue

            ## [NOTE] these have been generated
            if 'passages' in args.tag:
                if 'min' in args.tag:
                    selected = [int(psgid.split('#')[-1]) for psgid in qrels[example_id]]
                    ratings = ratings[selected] # selected
                    passages = [passages[s] for s in selected]

                outputs['num_segs'].append(len(passages))
                ratings = np.max(ratings[:args.topk], axis=0)
                context = " ".join(passages)
            else:
                ratings = np.array(judgements_all[example_id])
                outputs['num_segs'].append(len(contexts_all[example_id]))
                context = " ".join(contexts_all[example_id])

            ## calculate k-coverage
            n_correct = sum(ratings >= args.threshold)
            coverage = n_correct / len(questions)
            outputs['coverage'].append(coverage)

            ## calculate k-density
            n_tokens = len(tokenizer.tokenize(context))
            density = coverage / n_tokens
            outputs['density'].append(density)
            outputs['num_tokens'].append(n_tokens)

    # results
    mean_coverage = np.mean(outputs['coverage'])
    mean_density = np.mean(outputs['density'])
    mean_num_segments = np.mean(outputs['num_segs'])
    mean_num_tokens = np.mean(outputs['num_tokens'])
    logger.info(f'==== Evaluation Results ====')
    logger.info(f" # TAG : {args.tag.replace('topk', str(args.topk))} | Judge-LLM: {judge_LLM} | {len(outputs['coverage'])} examples")
    logger.info(f' # Mean Coverage(tau={args.threshold}) : {mean_coverage}')
    logger.info(f' # Mean Density (tau={args.threshold}) : {mean_density}')
    logger.info(f' # Mean number of segments : {mean_num_segments}')
    logger.info(f' # Mean number of tokens : {mean_num_tokens}\n')
