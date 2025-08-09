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

# Define the prompt for rating generation
# The one-shot generation is taken from NeuCLIR'23 report generation taask
prompt_template = """\
Instruction:
Create a statement of report request that corresponds to given report. The request should have clear information needs. Write the report request of approximately 100 words within <r> and </r> tags.

Report: Whether you dismiss UFOs as a fantasy or believe that extraterrestrials are visiting the Earth and flying rings around our most sophisticated aircraft, the U.S. government has been taking them seriously for quite some time. “Project Blue Book”, commissioned by the U.S. Air Force, studied reports of “flying saucers” but closed down in 1969 with a conclusion that they did not present a threat to the country. As the years went by UFO reports continued to be made and from 2007 to 2012 the Aerospace Threat Identification Program, set up under the sponsorship of Senator Harry Reid, spent $22 million looking into the issue once again. Later, the Pentagon formed a “working group for the study of unidentified aerial phenomena”. This study, staffed with personnel from Naval Intelligence, was not aimed at finding extraterrestrials, but rather at determining whether craft were being flown by potential U.S. opponents with new technologies. In June, 2022, in a report issued by the Office of the Director for National Intelligence and based on the observations made by members of the U.S. military and intelligence  from 2004 to 2021 it was stated that at that time there was, with one exception, not enough information to explain the 144 cases of what were renamed as “Unidentified Aerial Phenomena” examined. 

Report Request: <r>Produce a report on investigations within the United States in either the public or private sector into Unidentified Flying Objects (UFOs). The report should cover only investigative activities into still unidentified phenomena, and not the phenomena themselves. It should include information on the histories, costs, goals, and results of such investigations.</r>

Report: {report}

Report Request: <r>"""

def main(
    args,
    dataset='mds',
    subset='multi_news',
    load_mode='vllm',
    split='test'
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
        f"requests.{args.model.split('/')[-1]}.{args.shard}-{args.total_shards}.jsonl"
    )
    writer = open(output_path, "w")

    ## Prepare prompts
    prompts = [prompt_template.format(
        report=normalize_text(all_reports[id])
    ) for id in qids]

    # Start generation
    requests = []
    for batch_prompt in tqdm(
        batch_iterator(prompts, args.batch_size), 
        desc=f"Dataset: {dataset} (shard: {args.shard}/{args.total_shards})",
        total=len(prompts) // args.batch_size + 1
    ):
        output = llm.generate(batch_prompt, max_tokens=args.max_new_tokens)
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
        split='test'
    )
