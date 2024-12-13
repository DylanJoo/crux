import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import yaml
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm

from llm.base import LLM
from prompts.mds import *
from data_augmentation.utils import normalize_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--shard", type=int, default=0, help="the n-th shard")
    parser.add_argument("--shard_size", type=int, default=200, help="size of one shard")

    # Evaluation file is a json file that contains a list of item, each of which contains
    parser.add_argument("--multi_news_file", type=str, help="Path to multi-news")
    parser.add_argument("--wcep_10_file", type=str, help="Path to wcep-10")
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")

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
    parser.add_argument("--load_mode", type=str, default='no', help="Model to use")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")

    # Use summarization/extraction of the documents
    parser.add_argument("--ampere_gpu", default=False, action='store_true')
    # parser.add_argument("--used_field_in_demo", type=str, default=None, help="Use compressed text data. Option: `full`, `summary`, `extraction`")

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
    llm = LLM(args)
    
    # Generate prompts
    np.random.seed(args.seed)

    # Load training data
    train_data = None

    # Load evaluation data
    from datasets import load_from_disk, concatenate_datasets
    multi_news = load_from_disk(args.multi_news_file)['train'] # hf data
    wcep_10 = load_from_disk(args.wcep_10_file)['train'] # torch files are from original raw file

    ## super long document: multi_news 16430

    ## preproces documents here
    import re
    def normalize(string):
        string = string.strip()
        pattern = re.compile(r"\s+")
        string = re.sub(pattern, ' ', string).strip()
        pattern = re.compile(r"\n")
        string = re.sub(pattern, ' ', string).strip()
        pattern = re.compile("</s>")
        string = re.sub(pattern, '|||||', string).strip() # align seperation 
        return string

    multi_news = multi_news.map(lambda x: {"document": normalize(x['document']), 'mds-source': 'multi_news'})
    wcep_10 = wcep_10.map(lambda x: {"document": normalize(x['document']), 'mds-source': 'wcep-10'})
    eval_dataset = concatenate_datasets([multi_news, wcep_10])

    # Sample quick test
    if args.quick_test is not None:
        np.random.seed(args.seed)
        eval_ids = np.random.choice(len(eval_dataset), args.quick_test, replace=False)
        eval_dataset = [eval_dataset[int(idx)] for idx in eval_ids]

    # Generate the demonstration part
    if args.shot > 0:
        pass
    else:
        demo_prompt = ""


    # Generate the prompt
    eval_data = []
    logger.info("Generating prompts...") 
    for idx, eval_item in enumerate(tqdm(eval_dataset)):
        summary_text = normalize_texts(eval_item['summary'])
        prompt = prompt_statement_gen(
            INST=instruction_statement,
            D=summary_text,
            PREFIX="Statements:\n[1]: "
        )
        eval_data.append({
            'example_id': f"{eval_item['mds-source']}-{eval_ids[idx]}", 
            'shard_id': f"{args.shard}-{idx}", 
            'prompt': prompt,
            'full_text': eval_item['summary'],
            'ndoc': len(eval_item['document'].split('|||||'))
        })
    logger.info("Done prompt preparation.")

    # Start generation
    logger.info("Generating output...")
    start = args.shard * args.shard_size
    end = start + args.shard_size
    if start >= len(eval_data):
        exit(0) # finished

    eval_data = eval_data[start:end]
    for idx, item in enumerate(tqdm(eval_data)):
        prompt = item['prompt']
        prompt_len = len(llm.tokenizer.tokenize(prompt))
        # full_text = item.pop('full_text')

        ## The other claims of documetns are not.
        output_array = [llm.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len))]
        item['prompt_len'] = prompt_len
        
        output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
        if output_array[-1].endswith("End."):
            output_array[-1] = output_array[-1][:-len("End.")]

        logger.info(f"Example: {item['example_id']} -- {item['shard_id']}")
        logger.info(f"prompt text (length={prompt_len}): {prompt}")
        logger.info(f"Final model output: {output_array[-1]}") 

        ### if we dont have a good-ish summary's claim. we move on to the next
        if (len(output_array[-1].split('[')) < 2):
            item['output'] = []
            logger.info("Bypass this claims of documents', since the summary's is not successful")
        else:
            item['output'] = output_array if len(output_array) > 1 else output_array[0]
        
    # Save the result
    model_name = args.model
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    name = f"{args.dataset_name}-{model_name}-{args.shard}-{args.tag}-{args.seed}"

    if args.quick_test is not None:
        name += f"-susbet{args.quick_test}"

    eval_data = {"args": args.__dict__, "data": eval_data}

    if not os.path.exists("data/mds"):
        os.makedirs("data/mds")
    json.dump(eval_data, open("data/mds/" + name + ".json", "w"), indent=4)

if __name__ == "__main__":
    main()

