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

from augmentation.prompts import instruction_rating, prompt_rating_gen
from augmentation.utils import batch_iterator

def normalize_text(string):
    string = string.strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    return string

def replace_tags(sent, tag='q'):
    if tag == 'q':
        sent = re.sub(r"\<q\>|\<\/q\>", "\n", sent)
    if tag == 'p':
        sent = re.sub(r"\<p\>|\<\/p\>", "\n", sent)
    if tag == 't':
        sent = re.sub(r"\<r\>|\<\/r\>", "\n", sent)
    pattern = re.compile(r"\n+")
    sent = re.sub(pattern, '\n', sent)
    pattern = re.compile(r"^(\d+)*\.")
    sent = re.sub(pattern, '', sent)
    return sent

def load_question(path, n=10):
    data = json.load(open(path, 'r'))

    questions = []
    for i, item in enumerate(data['data']):
        example_id = item['example_id']
        outputs = item['output'].strip().split('</q>')[:n]
        outputs = [replace_tags(o, 'q').strip() for o in outputs]
        questions.append({"example_id": example_id, "texts": outputs})
    return questions

def load_passages(path, n=3):
    data = json.load(open(path, 'r'))

    passages = []
    for i, item in enumerate(data['data']):
        example_id = item['example_id']

        outputs = []
        for gen_output in item['docs']['output']:
            gen_output = normalize_text(gen_output)
            if gen_output == " ":
                outputs.append(["No content."])
            else:
                gen_output = gen_output.strip().split('</p>')[:n]
                gen_output = [replace_tags(o, 'p').strip() for o in gen_output]
                gen_output = [o.strip() for o in gen_output if o.strip() != ""]
                outputs.append(gen_output)

        passages.append({
            "example_id": example_id, 
            "texts": outputs, 
            "docs_full_texts": [normalize_text(d) for d in item["docs"]["full_text"]]
        })
    return passages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--shard_dir", type=str, help="directory for the input source")

    # Source files were tranformed into `dataset` arrow object
    parser.add_argument("--split", type=str, default='train', help="Original split of datasets")
    parser.add_argument("--shard", type=int, default=0, help="the n-th shard")
    parser.add_argument("--shard_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")

    # Model and name
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)") # use shard here
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--model_tag", type=str, help="Tag of run (for saving)") 
    parser.add_argument("--load_mode", type=str, default='no', help="['vllm', 'api']")

    # Decoding
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for generation")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=8192, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")

    # Use summarization/extraction of the documents
    parser.add_argument("--ampere_gpu", default=False, action='store_true')
    parser.add_argument("--port", default='8000', type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--n_questions", default=10, type=int)

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

    if args.load_mode == "api":
        from llm.requester import API
        llm = API(args)

    # Load source data

    questions_all = []
    for file in tqdm(glob(os.path.join(args.shard_dir, f"ques-gen/*-{args.split}-*.json"))):
        questions = load_question(file, args.n_questions)
        questions_all += questions
    questions_all = {q['example_id']: q['texts'] for q in questions_all}

    passages_all = []
    for file in tqdm(glob(os.path.join(args.shard_dir, f"psgs-gen/*-{args.split}-*.json"))):
        passages = load_passages(file)
        passages_all += passages
    documents_all = {p['example_id']: p['docs_full_texts'] for p in passages_all}
    passages_all = {p['example_id']: p['texts'] for p in passages_all}
    logger.info(f"Number of examples: questions -- {len(questions_all)} | passages -- {len(passages_all)}") 

    # sanity check
    overlap = questions_all.keys() & passages_all.keys()
    questions_all = {k: v for k, v in questions_all.items() if k in overlap}
    passages_all = {k: v for k, v in passages_all.items() if k in overlap}
    documents_all = {k: v for k, v in documents_all.items() if k in overlap}

    logger.info(f"{len(questions_all)} examples remained...")


    # Sharding
    start = args.shard * (args.shard_size or 0)
    end = start + (args.shard_size or len(data))
    ids = list(questions_all.keys())
    if start >= len(ids):
        exit(0)

    ids = ids[start:end]

    # Start generation
    logger.info("Generating output...")

    ratings = []
    for id in tqdm(ids, total=len(ids)):

        questions = questions_all[id]
        documents = documents_all[id]
        passages_set = passages_all[id]

        if len(passages_set) == 0:
            continue

        output_array = []
        for i, passage_list in enumerate(passages_set):
            for j, passage in enumerate(passage_list):

                prompts = [
                    prompt_rating_gen(
                        INST=instruction_rating,
                        Q=question,
                        C=passage,
                        PREFIX="Rating:"
                    ) for question in questions
                ]

                outputs = []
                for batch_prompts in batch_iterator(prompts, args.batch_size):

                    if args.load_mode == 'api':
                        output = llm.generate(prompts, max_tokens=args.max_new_tokens, min_tokens=2)
                        prompt_len = llm.prompt_len
                    else:
                        # prompt_len = len(llm.tokenizer.tokenize(prompt)) 
                        output = llm.generate(prompts, max_tokens=args.max_new_tokens, min_tokens=2)
                        prompt_len = -1

                    # extract rating
                    pattern = re.compile(r"\d|-\d")
                    output = [re.findall(pattern, o + "-1")[0] for o in output]
                    output = [-1 if len(o) == 0 else int(o) for o in output]
                    outputs += output

                output_array.append(outputs)

            logger.info(f"Example: {id} - doc #{i} (generated passages)")
            logger.info(f"Final model output: {output_array}") 

        ratings.append({
            "example_id": id,
            "documents": documents,
            "questions": questions,
            "passages": passages_set,
            "ratings": output_array
        })
        del output, output_array

    # Save the result
    output_dir = os.path.join(args.shard_dir, args.tag)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{args.model_tag}-{args.split}-{args.shard}.json")

    with open(output_file, "w") as f:
        for rating in ratings:
            f.write(json.dumps(rating)+'\n')

if __name__ == "__main__":
    main()

