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
import importlib

from ..tools import (
    batch_iterator, 
    load_corpus,
    load_run_or_qrel,
    load_ratings
)

# Define the prompt for rating generation
prompt_template = """\
Instruction:
Determine whether the question can be answered based on the provided context. Rate the context on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating.

Guideline: 
- 5: The context is highly relevant, complete, and accurate to the question.
- 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies to the question.
- 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies to the question.
- 2: The context has limited relevance and completeness, with significant gaps or inaccuracies to the question.
- 1: The context is minimally relevant or complete, with substantial shortcomings to the question.
- 0: The context is not relevant or complete at all.

Question: {question}
Context: {context} 

Rating: """

def main(
    args,
    dataset='researchy',
    subset=None,
    load_mode='vllm',
    split='test'
):

    # Load data-dependent functions 
    ir_utils = importlib.import_module(f"crux.tools.{dataset}.ir_utils", package=__name__)
    all_topic = ir_utils.load_topic() if subset is None else ir_utils.load_topic(subset=subset)
    all_subquestions = ir_utils.load_subtopics() if subset is None else ir_utils.load_subtopics(subset=subset)
    run = load_run_or_qrel(args.run_path, topk=args.top_k, threshold=3, threshold_score=0.6)
    qrel = ir_utils.get_qrel()
    corpus = load_corpus(args.corpus)
    # all_reports = ir_utils.load_report(subset=dataset, split=split)

    # Load the model or setup the API

    # Shard by topic (qid)
    qids = list(all_topic.keys())
    if (args.total_shards is not None) and (args.total_shards > 1):
        qids = sorted(qids)
        shard_size = len(qids) // args.total_shards + 1
        qids = qids[args.shard * shard_size: (args.shard + 1) * shard_size]

    # Ignore already done
    output_path = os.path.join(
        args.output_dir, 
        f"ratings.{args.model.split('/')[-1]}.qrel.jsonl"
    )

    # filter 1: existing qrels
    if os.path.exists(output_path.replace("-offload-qrel", "")): # remove the offload for original dataset
        ratings_done = load_ratings(output_path.replace("-offload-qrel", ""))
    else:
        ratings_done = {id: {} for id in qids}

    # Start generation
    for id in tqdm(qids, total=len(qids), desc=f"Dataset: {dataset} (shard: {args.shard}/{args.total_shards})"):

        subquestions = all_subquestions[id]
        doc_id_list = [docid for docid in qrel[id] if (docid not in ratings_done[id] and docid not in run[id])]
        doc_text_list = []
        for docid in doc_id_list:
            try:
                doc_text_list.append(corpus[docid])
            except:
                print(f"Document {docid} not found in corpus for id {id}. Skipping.")

        output_array = []
        for docid, doc in zip(doc_id_list, doc_text_list):

            for j, question in enumerate(subquestions):

                prompt = prompt_template.format(
                    question=question,
                    context=" ".join(doc['text'].split()[:2048])
                )
                message = [{"role": "user", "content": prompt}]

                with open(output_path, "a") as writer:
                    writer.write(json.dumps({"id": f"{id}::{docid}::{j}", "messages": message}) + "\n")

    logger.info(f"Ratings saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")

    parser.add_argument("--dataset", type=str, default="researchy", help="Dataset to use")
    parser.add_argument("--corpus", type=str, default=None, help="Path to the jsonl corpus file")
    parser.add_argument("--output_dir", type=str, default="./", help="Tag for the model")
    parser.add_argument("--run_path", type=str, default=None, help="Path to the run file (e.g., run.jsonl or qrel.jsonl)")
    parser.add_argument("--shard", type=int, default=0, help="the n-th shard")
    parser.add_argument("--total_shards", type=int, default=None, help="Total number of shards to split the dataset into")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k documents to consider for each query")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")

    # Model and decoding
    parser.add_argument("--load_mode", type=str, default='no', help="['vllm', 'api']")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_model_length", type=int, default=8192, help="Max length the model can take.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for generation")

    # Load config
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    print("Arguments")
    for k in args.__dict__:
        print(f"  {k}: {args.__dict__[k]}")

    os.makedirs(args.output_dir, exist_ok=True)
    main(
        args=args,
        dataset=args.dataset,
        load_mode=args.load_mode,
        split='test'
    )
