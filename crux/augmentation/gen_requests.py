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
    normalize_text,
    load_ratings
)

# Define the prompt for rating generation
# The one-shot generation is taken from NeuCLIR'23 report generation taask
prompt_template = """\
Instruction:
Create a statement of report request that corresponds to all the given sub-questions. The request should have clear information needs of sub-questions. Write the report request of approximately 100 words within <r> and </r> tags.

Sub-questions:
- What investigations into Unidentified Flying Objects (UFOs) or still unidentified phenomena have taken place within the United States?
- Which organizations or agencies conducted these investigations?
- What was the time period or duration of each investigation?
- What were the stated goals or objectives of each investigation?
- Were the investigations focused on determining potential extraterrestrial origins, national security threats, or other explanations?
- What were the budgets or financial costs associated with each investigation?
- Who funded these investigations, and through what sources or agencies?
- What were the key findings or conclusions of each investigation?
- Were any of the studied phenomena explained or identified as known objects or events?
- What proportion of the investigated cases remained unexplained?
- How has the U.S. approach to investigating UFOs or unidentified aerial phenomena changed over time?

Report request: <r>Produce a report on investigations within the United States in either the public or private sector into Unidentified Flying Objects (UFOs). The report should cover only investigative activities into still unidentified phenomena, and not the phenomena themselves. It should include information on the histories, costs, goals, and results of such investigations.</r>

Sub-questions:
{subquestions}

Report request: <r>"""

def merge_subquestions(questions):
    questions = [normalize_text(q) for q in questions]
    output = "\n- ".join(questions)
    output = "- " + output
    return output

def main(
    args,
    dataset='mds',
    subset='multi_news',
    load_mode='vllm',
    split='test',
):

    # Load data-dependent functions 
    ir_utils = importlib.import_module(f"crux.tools.{dataset}.ir_utils", package=__name__)
    all_topic = ir_utils.load_topic(split=split)
    all_subquestions = ir_utils.load_subtopics(split=split)

    # Filter unsanswerable questions
    all_subquestions_filterd = {}
    ratings = load_ratings(args.output_dir.replace('topic', 'judge'))
    for qid in all_subquestions:
        subquestions = all_subquestions[qid]
        answerable = (np.array([ratings[qid][docid] for docid in ratings[qid]]).max(0) >= 3).tolist()
        all_subquestions_filterd[qid] = [q for q, ans in zip(subquestions, answerable) if ans is True]

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
    qids = list(all_subquestions_filterd.keys())
    if (args.total_shards is not None) and (args.total_shards > 1):
        qids = sorted(qids)
        shard_size = len(qids) // args.total_shards + 1
        qids = qids[args.shard * shard_size: (args.shard + 1) * shard_size]

    output_path = os.path.join(
        args.output_dir, 
        f"requests.{split}.{args.model.split('/')[-1]}.{args.shard}-{args.total_shards}.jsonl"
    )
    writer = open(output_path, "w")

    ## Prepare prompts
    if args.add_main_query:
        prompts = [prompt_template.format(
            subquestions=merge_subquestions([all_topic[id]] + all_subquestions_filterd[id]) 
        ) for id in qids]
    else:
        prompts = [prompt_template.format(
            subquestions=merge_subquestions(all_subquestions_filterd[id]) 
        ) for id in qids]

    # Start generation
    requests = []
    for batch_prompt in tqdm(
        batch_iterator(prompts, args.batch_size), 
        desc=f"Dataset: {dataset} (shard: {args.shard}/{args.total_shards})",
        total=len(prompts) // args.batch_size + 1
    ):
        output = llm.generate(batch_prompt)
        output = [o.split('Instruction:')[0] for o in output]
        output = [o.split('</r>')[0] for o in output]
        output = [re.sub(r'<r>', '', o) for o in output]
        requests.extend(output)

    # Write output
    for id, request in zip(qids, requests):
        item = {"id": id, "request": request.strip()}
        writer.write(json.dumps(item) + "\n")

    output_dir = os.path.join(args.output_dir, args.tag)
    writer.close()
    logger.info(f"Requests saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")

    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset file")
    parser.add_argument("--subset", type=str, default=None, help="Subset of the dataset to use")
    parser.add_argument("--output_dir", type=str, help="directory for the output result")
    parser.add_argument("--shard", type=int, default=0, help="the n-th shard")
    parser.add_argument("--total_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--split", type=str, default='train', help="split of the datasets")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--add_main_query", action='store_true', default=False)

    # Model and decoding
    parser.add_argument("--load_mode", type=str, default='no', help="['vllm', 'api']")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max number of new tokens to generate in one step")
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
        split=args.split,
    )
