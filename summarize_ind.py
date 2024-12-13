import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import torch
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

from tools.ra_utils import (
    batch_iterator, 
    load_model, 
    load_run, 
    load_passages, 
    load_topics
)
def truncate_and_concat(texts, tokenizer, max_length=512, offset=6):
    tokenized = tokenizer.tokenize(texts)
    length = len(tokenizer.tokenize(texts))
    max_length = (max_length or tokenizer.max_len_single_sentence-1)
    if (length+offset) < max_length:
        return texts
    else:
        return tokenizer.convert_tokens_to_string(tokenized[:(max_length-6)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--model_class", type=str, default=None, choices=["fid", "seq2seq", "causalLM", "vllm"])
    parser.add_argument("--template", type=str, default="title: {T} content: {P}")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--query_focus", action='store_true', default=False)

    parser.add_argument("--run_file", type=str, default=None)
    parser.add_argument("--topk", type=int, default=9999)
    parser.add_argument("--topic_file", type=str, default=None)
    parser.add_argument("--passage_dir", type=str, default=None)
    parser.add_argument("--do_irrelevant", action='store_true', default=False)

    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--truncate", type=int, default=6)
    args = parser.parse_args()

    # load model
    model, tokenizer = load_model(args.model_name_or_path, model_class=args.model_class.lower())

    # load run, topics, passages
    run = load_run(args.run_file, args.topk)
    topics = load_topics(args.topic_file)
    passages = load_passages(args.passage_dir)
    # load previous results (to avoid rerun)
    summaries = load_passages(args.output_file)

    psgids = []
    input_data = []
    for example_id in run:
        topic = topics[example_id]
        candidate_list = run[example_id]

        for psgid in candidate_list:
            psg = passages[psgid]

            # run only if the query-passage has not been done and is relevant pasages
            if (example_id == psgid.split(':')[0]) and (psgid not in summaries):
                psg = truncate_and_concat(
                    psg,
                    tokenizer=tokenizer,
                    max_length=args.max_length, 
                    offset=args.truncate
                )
                input_data.append({
                    'example_id': example_id, 'pid': psgid, 
                    'input': args.template.replace("{Q}", topic).replace("{P}", psg)
                })

            elif (example_id != psgid.split(':')[0]) and (args.do_irrelevant) and (psgid not in summaries):
                psg = truncate_and_concat(
                    psg,
                    tokenizer=tokenizer,
                    max_length=args.max_length, 
                    offset=args.truncate
                )
                if psgid not in psgids:
                    input_data.append({
                        'example_id': example_id, 'pid': psgid, 
                        'input': args.template.replace("{Q}", topic).replace("{P}", psg)
                    })
                    psgids.append(psgid)

    # generate summaries
    logger.info(f'Existing summaries have {len(summaries)} summaries.')
    logger.info(f'Generate summaries for total {len(input_data)} passages.')

    for batch_input_data in tqdm(
        batch_iterator(input_data, args.batch_size), desc=f'{args.model_class} summarization'
    ):

        batch_inputs = [b.pop('input') for b in batch_input_data]

        if args.model_class == 'vllm':
            outputs = model.generate(batch_inputs, max_new_tokens=512)
            outputs = [o.split('</s>')[0] for o in outputs]
        else:
            tokenized_input = tokenizer(
                batch_inputs, 
                padding=True, 
                truncation=True, 
                max_length=args.max_length, 
                return_tensors='pt'
            ).to(model.device)
            outputs = model.generate(**tokenized_input, min_new_tokens=32, max_new_tokens=512)
            outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

        for i, output in enumerate(outputs):
            psgid = batch_input_data[i]['pid']
            summaries[psgid] = output

    # rewrite
    with open(args.output_file, 'w') as f:
        for psgid in summaries:
            summary = summaries[psgid]
            f.write(json.dumps({"id": psgid, "contents": summary}, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    with torch.no_grad():
        main()

