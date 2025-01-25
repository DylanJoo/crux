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
from datasets import load_from_disk

from augmentation.prompts import instruction_request, prompt_topic_gen, demo
from augmentation.utils import batch_iterator

def normalize_list(string_list):
    for i in range(len(string_list)):
        string_list[i] = normalize_text(string_list[i])
    return string_list

def flatten_and_normalize(string_list):
    string = " ".join(string_list)
    return normalize_text(string)

def normalize(string):
    string = string.strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile("</s>")
    string = re.sub(pattern, '|||||', string).strip() # align seperation 
    return string.split('|||||')

def normalize_text(string):
    string = string.strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    return string

def postprocess(output):
    output = output.replace("<|im_end|>", "").rstrip()
    if output.endswith("End."):
        output = output[:-len("End.")]
    output = output.split('Note:')[0]
    output = output.split('Report:')[0]
    output = output.split('Report request:')[0]
    output = output.split('Instruction:')[0]
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--output_dir", type=str, help="directory for the output result")

    # Source files were tranformed into `dataset` arrow object
    parser.add_argument("--multi_news_file", type=str, default=None)
    parser.add_argument("--duc04_file", type=str, default=None)
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
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=8192, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")

    # Use summarization/extraction of the documents
    parser.add_argument("--ampere_gpu", default=False, action='store_true')
    parser.add_argument("--port", default='8000', type=str)
    parser.add_argument("--num_gpus", default=1, type=int)

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
    
    # Set seed
    np.random.seed(args.seed)

    # Load source data
    ## Preprocess

    if args.multi_news_file is not None:
        multi_news = load_from_disk(args.multi_news_file)[args.split]
        multi_news = multi_news.map(lambda x: {
            "document": normalize(x['document']), 
            'mds-source': 'multi_news'
        })
        multi_news = multi_news.filter(lambda x: len(x['document']) >=2 )
        # [No need to chunk documents]
        dataset = multi_news

    if args.duc04_file is not None:
        duc04 = load_from_disk(args.duc04_file)['train']
        duc04 = duc04.map(lambda x: {
            "document": normalize_list(x['context']),
            "summary": flatten_and_normalize(x['summary']),
            'mds-source': 'duc04'
        })
        duc04 = duc04.filter(lambda x: len(x['document']) >=2 )
        # [No need to chunk documents]
        dataset = duc04

    ## Sample
    if args.split == 'train':
        dataset = [dataset[idx] for idx in range(len(dataset))]
    else:
        dataset = [dataset[idx] for idx in range(min(5000, len(dataset)))]
    ids = list(range(len(dataset)))

    # Generate prompts # [NOTE] the documents may be shorter and more than the original document.
    n_total = 0
    data = []
    logger.info(f"Length of dataset: {len(dataset)}") 
    for idx, item in enumerate(tqdm(dataset)):
        document_list = item['document']
        summary_text = normalize_text(item['summary'])

        prompt = prompt_topic_gen(
            INST=instruction_request,
            DEMO=demo,
            D=summary_text,
            PREFIX="Report request: <r>"
        )

        data.append({
            'example_id': f"{item['mds-source']}-{args.split}-{ids[idx]}", 
            'shard_id': f"{args.shard}-{idx}", 
            'summary': summary_text,
            'prompt': prompt,
        })
        n_total += len(document_list)
    logger.info(f"Total number of prompts: {len(data)} | {n_total}")

    # Sharding
    start = args.shard * (args.shard_size or 0)
    end = start + (args.shard_size or len(data))
    if start >= len(data):
        exit(0)

    data = data[start:end]

    # Start generation
    logger.info("Generating output...")

    for items in tqdm(
        batch_iterator(data, size=args.batch_size), total=len(data)//args.batch_size+1
    ):

        prompt = [item['prompt'] for item in items]
        if args.load_mode == 'api':
            output = llm.generate(prompt[0], max_tokens=args.max_new_tokens)
            prompt_len = llm.prompt_len
        else:
            # prompt_len = len(llm.tokenizer.tokenize(prompt)) 
            output = llm.generate(prompt, max_tokens=args.max_new_tokens)
            prompt_len = -1

        logger.info(f"Example: {items[-1]['example_id']} -- {items[-1]['shard_id']}")
        logger.info(f"prompt text (length={prompt_len}): {prompt[-1]}")
        logger.info(f"Final model output: {output[-1]}") 

        for i, item in enumerate(items):
            item['output'] = postprocess(output[i])

    # Save the result
    data = {"args": args.__dict__, "data": data}

    output_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{args.model_tag}-{args.split}-{args.shard}.json")
    json.dump(data, open(output_file, 'w'), indent=4)

if __name__ == "__main__":
    main()

