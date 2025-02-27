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
    pattern = re.compile(r"\n+")
    sent = re.sub(pattern, '\n', sent)
    return sent

def load_question(path, n=10):
    data = json.load(open(path, 'r'))

    questions = []
    for i, item in enumerate(data['data']):
        example_id = item['example_id']
        if not isinstance(item['output'], list):
            outputs = item['output'].strip().split('</q>')[:n]
            outputs = [replace_tags(o).strip() for o in outputs]
            questions.append({"example_id": example_id, "texts": outputs})
    return questions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--shard_dir", type=str, help="Path to pre-generated results")
    parser.add_argument("--context_file", type=str, help="Path to retrieval-augmented context")
    parser.add_argument("--output_file", type=str, help="Path to judged (graded) retrieval-augmented context")
    parser.add_argument("--topics", type=str, help="Path to topic query")

    # Evaluation file is a json file that contains a list of item, each of which contains
    parser.add_argument("--split", type=str, default='train', help="Original split of datasets")

    # Model and name
    # parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
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
    parser.add_argument("--port", default='8000', type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--n_questions", default=10, type=int)
    parser.add_argument("--quant", default=None, type=str)
    parser.add_argument("--top_k", default=-1, type=int)

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

    logger.info("load questions...") 
    questions_all = []
    for file in tqdm(glob(os.path.join(args.shard_dir, f"ques-gen/*-{args.split}-*.json"))):
        questions = load_question(file, args.n_questions)
        questions_all += questions
    questions_all = {q['example_id']: q['texts'] for q in questions_all}

    logger.info("load topics...") 
    topics_all = {}
    with open(args.topics) as f:
        for line in f:
            id, text = line.split('\t')
            topics_all[id] = text.strip()

    logger.info("load context...") 
    context_type = ""
    contexts_all = {}
    with open(args.context_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if not isinstance(data['output'], list):
                contexts_all[data['example_id']] = [data['output']]
            else:
                contexts_all[data['example_id']] = data['output']
    context_type = data['type']
    logger.info(f"load retrieval context {context_type}...") 

    # Start generation
    logger.info("Generating output...")

    ratings = []
    for t, example_id in enumerate(tqdm(topics_all)):
        questions = questions_all[example_id]
        topic = topics_all[example_id]
        context_list = contexts_all[example_id]

        output = ""
        output_array = []
        for i, context in enumerate(context_list[:args.top_k]):

            output_vector = [-1 for _ in questions]
            for k, question in enumerate(questions):
                prompt = prompt_rating_gen(
                    INST=instruction_rating,
                    Q=question,
                    C=context,
                    PREFIX="Rating:"
                )
                output = llm.generate(prompt, 
                    max_tokens=args.max_new_tokens,
                    min_tokens=1
                )
                output = output.replace("<|im_end|>", "").rstrip()
                if output.endswith("End."):
                    output = output[:-len("End.")]

                # extract rating
                pattern = re.compile(r"\d|-\d")
                output = re.findall(pattern, output + "-1")[0]
                output = -1 if len(output) == 0 else int(output)
                output_vector[k] = output

            output_array.append(output_vector)

        ## aggregate the ratings of retrieval context
        max_output_array = np.max(output_array, axis=0).tolist()
        logger.info(f"Example: {example_id} | #Context: {i+1}")
        logger.info(f"Final model output: {output_vector}") 

        ratings.append({
            "example_id": example_id,
            "topic": topic, 
            "questions": questions,
            "judge_LLM": args.model_tag,
            "contexts": context_list,
            "judgements": max_output_array,
            "judgement_all": output_array
        })

    # Save the result
    output_file = args.output_file.replace('outputs', 'judgements')

    with open(output_file, "w") as f:
        for rating in ratings:
            f.write(json.dumps(rating)+'\n')

if __name__ == "__main__":
    main()

