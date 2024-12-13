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

def normalize_text(string):
    string = string.strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    return string

def load_passages(path, n=3):
    data = json.load(open(path, 'r'))

    passages = []
    for i, item in enumerate(data['data']):
        example_id = item['example_id']

        doc_outputs = []
        for doc_output in item['docs']['output']:
            doc_output = normalize_text(doc_output)
            if doc_output == " ":
                doc_outputs.append(["No content."])
            else:
                doc_output = doc_output.strip().split('</p>')[:n]
                doc_output = [replace_tags(o, 'p').strip() for o in doc_output]
                doc_output = [o.strip() for o in doc_output if o.strip() != ""]
                doc_outputs.append(doc_output)

        passages.append({
            "example_id": example_id, 
            "texts": doc_outputs, 
            "docs_full_texts": [normalize_text(d) for d in item["docs"]["full_text"]]
        })
    return passages

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
    parser.add_argument("--output_dir", type=str, help="Path to generated results")

    # Evaluation file is a json file that contains a list of item, each of which contains
    parser.add_argument("--split", type=str, default='train', help="Original split of datasets")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=1000)

    # ICL setting
    parser.add_argument("--ndoc", type=int, help="Number of documents, the exact number will go in decoder.")
    parser.add_argument("--ndoc_pool", type=None, help="Number of documents pool. None will be the same as ndoc")
    parser.add_argument("--shot", type=int, help="Number of ICL demonstrations")
    parser.add_argument("--ndoc_in_demo", type=int, default=None, help="When using --fewer_doc_in_demo, use this to designate how many docs in demo")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")

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


    logger.info("load questions...") 
    questions_all = []
    for file in tqdm(glob(os.path.join(args.shard_dir, f"ques-gen/*-{args.split}-*.json"))):
        questions = load_question(file, args.n_questions)
        questions_all += questions
    questions_all = {q['example_id']: q['texts'] for q in questions_all}

    logger.info("load passages (and documents)...") 
    passages_all = []
    for file in tqdm(glob(os.path.join(args.shard_dir, f"psgs-gen/*-{args.split}-*.json"))):
        passages = load_passages(file)
        passages_all += passages
    documents_all = {p['example_id']: p['docs_full_texts'] for p in passages_all}
    passages_all = {p['example_id']: p['texts'] for p in passages_all}
    logger.info(f"Number of examples: questions -- {len(questions_all)} | passages -- {len(passages_all)}") 

    # get intersection
    overlap = questions_all.keys() & passages_all.keys()
    questions_all = {k: v for k, v in questions_all.items() if k in overlap}
    passages_all = {k: v for k, v in passages_all.items() if k in overlap}
    documents_all = {k: v for k, v in documents_all.items() if k in overlap}
    logger.info(f"{len(questions_all)} examples remained...")

    # Start generation
    logger.info("Generating output...")

    # Sharding
    if args.shard is not None:
        start = args.shard * args.shard_size
        end = min( (args.shard+1) * args.shard_size, len(questions_all) )
        ids = list(questions_all.keys())
        ids = ids[start:end]
    else:
        ids = list(questions_all.keys())

    ratings = []
    for example_id in tqdm(ids):
        questions = questions_all[example_id]
        documents = documents_all[example_id]
        passages_set = passages_all[example_id]

        output = ""
        output_array = []
        if len(passages_set) == 0:
            continue

        for i, passage_list in enumerate(passages_set):
            for j, passage in enumerate(passage_list):

                output_vector = [-1 for _ in questions]
                for k, question in enumerate(questions):
                    prompt = prompt_rating_gen(
                        INST=instruction_rating,
                        Q=question,
                        C=passage,
                        PREFIX="Rating:"
                    )
                    if args.load_mode == 'api':
                        output = llm.generate(prompt, max_tokens=args.max_new_tokens)
                        prompt_len = llm.prompt_len
                    else:
                        prompt_len = len(llm.tokenizer.tokenize(prompt))
                        output = llm.generate(prompt, 
                            max_tokens=min(args.max_new_tokens, args.max_length-prompt_len),
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

            logger.info(f"Example: {example_id} - doc #{i} (generated passages)")
            logger.info(f"Final model output: {output_vector}") 

        ratings.append({
            "example_id": example_id,
            "documents": documents,
            "questions": questions,
            "passages": passages_set,
            "ratings": output_array
        })
        del output, output_array

    # Save the result
    output_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(output_dir, exist_ok=True)

    if args.shard is not None:
        output_file = os.path.join(output_dir, f"{args.model_tag}-{args.split}-0.jsonl")
    else:
        output_file = os.path.join(output_dir, f"{args.model_tag}-{args.split}.jsonl")
    with open(output_file, "w") as f:
        for rating in ratings:
            f.write(json.dumps(rating)+'\n')

if __name__ == "__main__":
    main()

