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
from evaluation.llm_judge.utils import (
    load_questions, 
    load_contexts, 
    load_judgements
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--topic_question_file", type=str, help="Path to retrieval-augmented context")
    parser.add_argument("--output_file", type=str, help="Path to judged (graded) retrieval-augmented context")
    parser.add_argument("--context_file", type=str, help="Path to contexts")

    # Evaluation file is a json file that contains a list of item, each of which contains
    parser.add_argument("--split", type=str, default='train', help="Original split of datasets")

    # Model and name
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)") # use shard here
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--model_tag", type=str, help="Tag of run (for saving)") 
    parser.add_argument("--load_mode", type=str, default='no', help="['vllm', '8bit', '4bit', 'api']")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=16, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")

    # Use summarization/extraction of the documents
    parser.add_argument("--ampere_gpu", default=False, action='store_true')
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--n_questions", default=10, type=int)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--quant", default=None, type=str)

    # Load config
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    logger.info(f"Set the model max length to {args.max_length} (if not correct, check the code)")

    # Load the model or setup the API
    if args.load_mode == 'vllm':
        from llm.base import vLLM
        llm = vLLM(args)
    elif args.load_mode == "api":
        from llm.requester import API
        llm = API(args)
    else:
        from llm.base import LLM
        llm = LLM(args)

    # load questions, load_run
    questions_all = load_questions(args.topic_question_file, args.n_questions)
    contexts = load_contexts(args.context_file)

    judgements = load_judgements(args.output_file)
    logger.info(f"Total number of judged pairs {len(judgements)} / {len(contexts)}") 

    for psgid in tqdm(contexts, desc=f'Evaluating context ', total=len(contexts)):

        example_id = psgid.split(":")[0]

        # skip the one that have already been done
        if example_id not in questions_all:
            continue 

        questions = questions_all[example_id]
        context = contexts[psgid]

        output = ""
        output_vector = [-1 for _ in questions]

        # skip the one that have already been done
        if judgements[example_id][psgid] is not None:
            continue 

        ## no batch here
        for k, question in enumerate(questions):
            prompt = prompt_rating_gen(
                INST=instruction_rating,
                Q=question,
                C=context,
                PREFIX="Rating:"
            )
            output = llm.generate(
                prompt, 
                max_tokens=args.max_new_tokens,
                min_tokens=1
            )
            output = [o.replace("<|im_end|>", "").rstrip() for o in output][0]

            # extract rating
            pattern = re.compile(r"\d|-\d")
            output = re.findall(pattern, output + "-1")[0]
            output = -1 if len(output) == 0 else int(output)
            output_vector[k] = output

        # append on output array
        judgements[example_id][psgid] = output_vector
        logger.info(f"Final model output: {output_vector}") 

    # Save the result
    with open(args.output_file, "w") as f:
        for example_id in judgements:
            for psgid in judgements[example_id]:
                rating = judgements[example_id][psgid]
                f.write(json.dumps({"example_id": example_id, "pid": psgid, "rating": rating})+'\n')

if __name__ == "__main__":
    main()

