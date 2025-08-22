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
import pdb

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
    run = load_run_or_qrel(args.run_path, topk=args.top_k, threshold=1)
    corpus = load_corpus(args.corpus)
    # all_reports = ir_utils.load_report(subset=subset, split=split)

    # Load the model or setup the API
    if args.load_mode == 'litellm':
        from ..llm.litellm_api import LLM
    if args.load_mode == 'vllm':
        from ..llm.vllm_async import LLM

    llm = LLM(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        num_gpus=args.num_gpus
    )

    # Shard by topic (qid)
    qids = list(all_topic.keys())
    if (args.total_shards is not None) and (args.total_shards > 1):
        qids = sorted(qids)
        shard_size = len(qids) // args.total_shards + 1
        qids = qids[args.shard * shard_size: (args.shard + 1) * shard_size]

    output_path = os.path.join(
        args.output_dir, 
        f"ratings.{args.model.split('/')[-1]}.{args.shard}-{args.total_shards}.jsonl"
    )
    if os.path.exists(output_path):
        ratings_done = load_ratings(output_path)
        # TODO: remove this?
        # for id in ratings_done:
        #     hits = [docid for docid in run[id]]
        #     run[id] = {docid: run[id][docid] for docid in hits if docid not in ratings_done[id]}
        # ratings_done = {id: v for id, v in ratings_done.items() if id in qids}
    else:
        ratings_done = {id: {} for id in qids}

    # Start generation
    ratings = []
    for id in tqdm(qids, total=len(qids), desc=f"Dataset: {dataset} (shard: {args.shard}/{args.total_shards})"):

        subquestions = all_subquestions[id]
        doc_id_list = [docid for docid in run[id] if docid not in ratings_done[id]]
        doc_text_list = [corpus[docid] for docid in doc_id_list]

        output_array = []
        for i, doc in enumerate(doc_text_list):

            prompts = [
                prompt_template.format(
                    question=question,
                    context=" ".join(doc['text'].split()[:2048])  # Limit context to 2048 tokens
                ) for question in subquestions
            ]

            output_list = []
            for batch_prompts in batch_iterator(prompts, args.batch_size):
                output = llm.generate(batch_prompts)
                pattern = re.compile(r"\d|-\d")
                output = [re.findall(pattern, o + "-1")[0] for o in output]
                output = [-1 if len(o) == 0 else int(o) for o in output]
                output_list.extend(output)

            output_array.append(output_list)
            assert len(output_list) == len(subquestions), f"Mismatched length"

            if i % 200 == 0:
                logger.info(f"Processed example:\nPrompt: {prompts[0]}\nOutput: {output_list[0]}")

        # write output
        with open(output_path, "a") as writer:
            for i, (docid, doctext) in enumerate(zip(doc_id_list, doc_text_list)):
                item = {"id": id, "docid": docid, "rating": output_array[i]}
                ratings.append(item)
                writer.write(json.dumps(item) + "\n")

    logger.info(f"Ratings saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")

    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset file")
    parser.add_argument("--subset", type=str, default=None, help="Subset of the dataset to use")
    parser.add_argument("--output_dir", type=str, help="directory for the output result")
    parser.add_argument("--shard", type=int, default=0, help="the n-th shard")
    parser.add_argument("--total_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--corpus", type=str, default=None, help="Path to the jsonl corpus file")
    parser.add_argument("--run_path", type=str, default=None, help="Path to the run file (e.g., run.jsonl or qrel.jsonl)")
    parser.add_argument("--top_k", type=int, default=9999, help="Top-k documents to consider for each query")

    # Model and decoding
    parser.add_argument("--load_mode", type=str, default=None)
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Max length the model can take.")
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
        subset=args.subset,
        split='test'
    )
