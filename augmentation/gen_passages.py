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

from prompts.mds import *

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

def maybe_chunking(dlist, n=1024):
    overlength = [(i, len(d.split()) > n) for i, d in enumerate(dlist)]

    if any([o for _, o in overlength]):
        to_return = []
        for i, do_chunk in overlength:
            if do_chunk:
                words = dlist[i].split()
                while len(words) > 0:
                    to_return.append(" ".join(words[:512]))
                    words = words[512:]
            else:
                to_return.append(dlist[i])
        return to_return
    else:
        return dlist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--shard", type=int, default=0, help="the n-th shard")
    parser.add_argument("--shard_size", type=int, default=None, help="size of one shard")
    parser.add_argument("--output_dir", type=str, help="directory for the output result")

    # Evaluation file is a json file that contains a list of item, each of which contains
    parser.add_argument("--multi_news_file", type=str, default=None)
    parser.add_argument("--duc04_file", type=str, default=None)
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")
    parser.add_argument("--split", type=str, default='train', help="Original split of datasets")

    # ICL setting
    parser.add_argument("--ndoc", type=int, help="Number of documents, the exact number will go in decoder.")
    parser.add_argument("--ndoc_pool", type=None, help="Number of documents pool. None will be the same as ndoc")
    parser.add_argument("--shot", type=int, help="Number of ICL demonstrations")
    parser.add_argument("--ndoc_in_demo", type=int, default=None, help="When using --fewer_doc_in_demo, use this to designate how many docs in demo")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")

    # Model and name
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)") # use shard here
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--model_tag", type=str, help="Tag of run (for saving)") 
    parser.add_argument("--load_mode", type=str, default='no', help="['vllm', '8bit', '4bit', 'api']")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")

    # Use summarization/extraction of the documents
    parser.add_argument("--ampere_gpu", default=False, action='store_true')
    parser.add_argument("--port", default='8000', type=str)
    parser.add_argument("--num_gpus", default=1, type=int)

    # Load config
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    if "turbo" in args.model:
        args.max_length = 4096
    if "16k" in args.model:
        args.max_length = 16384
    elif "32k" in args.model:
        args.max_length = 32768
    elif "turbo" in args.model:
        args.max_length = 4096
    elif "gpt-4" in args.model:
        args.max_length = 8192
    elif "llama-2" in args.model.lower() or "llama2" in args.model.lower():
        args.max_length = 4096
    elif "llama-3" in args.model.lower() or "llama3" in args.model.lower():
        args.max_length = 8192
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    logger.info(f"Set the model max length to {args.max_length} (if not correct, check the code)")

    if args.ndoc_pool is None:
        args.ndoc_pool = args.ndoc
    logger.info(f"Set the model max number of documents to {args.ndoc}/{args.ndoc_pool}")
        
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
    
    # Generate prompts
    np.random.seed(args.seed)

    # Load training data
    train_data = None

    # Load evaluation data
    from datasets import load_from_disk
    if args.multi_news_file is not None:
        multi_news = load_from_disk(args.multi_news_file)[args.split]

        multi_news = multi_news.map(lambda x: {
            "document": normalize(x['document']), 
            'mds-source': 'multi_news'
        })
        multi_news = multi_news.filter(lambda x: len(x['document']) >=2 )
        multi_news = multi_news.map(lambda x: {
            "document": maybe_chunking(x['document'], n=1024)
        })
        dataset = multi_news

    if args.duc04_file is not None:
        duc04 = load_from_disk(args.duc04_file)['train']
        duc04 = duc04.map(lambda x: {
            "document": normalize_list(x['context']),
            "summary": flatten_and_normalize(x['summary']),
            'mds-source': 'duc04'
        })
        duc04 = duc04.filter(lambda x: len(x['document']) >=2 )
        duc04 = duc04.map(lambda x: {
            "document": maybe_chunking(x['document'], n=1024)
        })
        dataset = duc04

    # Sample quick test
    if args.quick_test is not None:
        np.random.seed(args.seed)
        ids = np.random.choice(len(dataset), args.quick_test, replace=False)
        dataset = [dataset[int(idx)] for idx in ids]
    else:
        if args.split == 'train':
            dataset = [dataset[idx] for idx in range(len(dataset))]
        else:
            dataset = [dataset[idx] for idx in range(min(5000, len(dataset)))]
        ids = list(range(len(dataset)))

    # Generate the prompt
    n_total = 0
    data = []
    logger.info(f"Length of dataset: {len(dataset)}") 
    logger.info("Generating prompts...") 
    for idx, item in enumerate(tqdm(dataset)):
        document_list = item['document'] # [NOTE] the documents may be shorter and more than the original document.

        prompt_list = []
        for document in document_list:
            prompt = prompt_passage_gen(
                INST=instruction_passage,
                D=document,
                PREFIX="Passages:\n<p>"
            )
            prompt_list.append(prompt)

        data.append({
            'example_id': f"{item['mds-source']}-{args.split}-{ids[idx]}", 
            'shard_id': f"{args.shard}-{idx}", 
            'summary': normalize_text(item['summary']),
            'ndoc': len(document_list),
            'docs': {'full_text': document_list, 'prompt': prompt_list }
        })
        n_total += len(document_list)
    logger.info(f"Done prompt preparation. Total number of prompts: {len(data)} | {n_total}")

    # Start generation
    logger.info("Generating output...")
    start = args.shard * (args.shard_size or 0)
    end = start + (args.shard_size or len(data))
    if start >= len(data):
        exit(0) # finished

    data = data[start:end]
    for idx, item in enumerate(tqdm(data, "augmenting", total=len(data))):
        output_array = []

        for prompt in item['docs']['prompt']:
            if args.load_mode == 'api':
                output = llm.generate(prompt, max_tokens=args.max_new_tokens)
                prompt_len = llm.prompt_len
            else:
                prompt_len = len(llm.tokenizer.tokenize(prompt))
                output = llm.generate(prompt, 
                    max_tokens=min(args.max_new_tokens, args.max_length-prompt_len),
                )

            ## postprocess for consistent format
            output = output.replace("<|im_end|>", "").rstrip()
            if output.endswith("End."):
                output = output[:-len("End.")]

            output = output.split('Note: ')[0]

            if output == "":
                logger.info(f"Original raw output: {output}")
                output = llm.generate(prompt, 
                    max_tokens=min(args.max_new_tokens, args.max_length-prompt_len), 
                    min_tokens=64
                )
            output_array.append(output)

        logger.info(f"Example: {item['example_id']} -- {item['shard_id']}")
        logger.info(f"prompt text (length={prompt_len}): {prompt}")
        logger.info(f"Final model output: {output_array[-1]}") 
        logger.info(f"Number of documents {item['ndoc']}") 
        item['docs']['output'] = output_array
        if idx != 0:
            item['docs']['prompt'] = ""

    # Save the result
    data = {"args": args.__dict__, "data": data}

    output_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{args.model_tag}-{args.split}-{args.shard}.json")
    json.dump(data, open(output_file, 'w'), indent=4)

if __name__ == "__main__":
    main()

