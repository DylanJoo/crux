import os
import sys
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
    normalize_text
)

prompt_template = """\
Instruction: 
Write {n_subquestions} important diverse sub-questions that can reveal the unique information contained in the given report. Each sub-question should be self-contained and have necessary context to understand. Each sub-qeustion should cover different aspects of the report. Write the sub-question within `<q>` and `</q>` tags. Do not include any other text or explanation.

Report: 
{report}

Sub-questions:
<q>"""

def main(
    args,
    dataset='mds',
    subset='multi_news',
    load_mode='vllm',
    split='test',
    n_subquestions=10,
):

    # Load data-dependent functions 
    ir_utils = importlib.import_module(f"crux.tools.{dataset}.ir_utils", package=__name__)
    # all_topic = ir_utils.load_topic()
    # all_subquestions = ir_utils.load_subtopics()
    # run = load_run_or_qrel(args.run_path, topk=20, threshold=3)
    # corpus = load_corpus(args.corpus)
    all_reports = ir_utils.load_reports(subset=subset, split=split)

    # Load the model or setup the API
    from ..llm.litellm_api import LLM
    llm = LLM(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        max_model_length=args.max_model_length,
    )

    # Shard by topic (qid)
    qids = list(all_reports.keys())
    if (args.total_shards is not None) and (args.total_shards > 1):
        qids = sorted(qids)
        shard_size = len(qids) // args.total_shards + 1
        qids = qids[args.shard * shard_size: (args.shard + 1) * shard_size]

    output_path = os.path.join(
        args.output_dir, 
        f"subquestions.{args.model.split('/')[-1]}.{args.shard}-{args.total_shards}.jsonl"
    )
    writer = open(output_path, "w")

    ## Prepare prompts
    prompts = [prompt_template.format(
        report=normalize_text(all_reports[id]),
        n_subquestions=args.n_subquestions
    ) for id in qids]

    # Start generation
    subquestions = []
    for batch_prompt in tqdm(
        batch_iterator(prompts, args.batch_size), 
        desc=f"Dataset: {dataset} (shard: {args.shard}/{args.total_shards})",
        total=len(prompts) // args.batch_size + 1
    ):
        output = llm.generate(batch_prompt, max_tokens=args.max_new_tokens)
        output = [o.split('Instruction:')[0] for o in output]
        output_processed = []
        for o in output:
            o = o.strip().split('</q>')[:args.n_subquestions]
            o = [re.sub(r'<q>', '', o) for o in o]
            o = [re.sub(r'(\s)*-\s', '', q).strip() for q in o]
            o = [re.sub(r'^\s*\d+[\.\)\-]?\s*', '', q).strip() for q in o]
            output_processed.append(o)

        subquestions.extend(output_processed)

    # Write output
    for id, subquestion_set in zip(qids, subquestions):
        item = {"id": id, "subquestions": subquestion_set}
        writer.write(json.dumps(item) + "\n")

    output_dir = os.path.join(args.output_dir, args.tag)
    writer.close()
    logger.info(f"Subquestions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")

    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset file")
    parser.add_argument("--subset", type=str, default=None, help="Subset of the dataset to use")
    parser.add_argument("--output_dir", type=str, help="directory for the output result")
    parser.add_argument("--shard", type=int, default=0, help="the n-th shard")
    parser.add_argument("--total_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")

    # Model and decoding
    parser.add_argument("--load_mode", type=str, default='no', help="['vllm', 'api']")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_model_length", type=int, default=8192, help="Max length the model can take.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for generation")
    parser.add_argument("--n_subquestions", type=int, default=10, help="Number of sub-questions to generate per report")

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
        subset=args.subset,
        load_mode=args.load_mode,
        split='test',
        n_subquestions=args.n_subquestions
    )
