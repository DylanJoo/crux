import os
import torch
import json
import argparse
from tqdm import tqdm
from utils import batch_iterator, load_model
from data import Standard, StandardWithPrefix

def generate_standard_with_prefix():
    pass

def generate_standard(
    request, 
    docs,
):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_key", type=str, default="summary_debug")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--template", type=str, default="title: {T} content: {P}")

    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name_or_path, model_class='fid')
    model.eval()

    eval_data = load_dataset('json', data_files=args.eval_file, keep_in_memory=True)['train']

    collator = Standard

    for eval_data_item in tqdm(eval_data, total=len(eval_data)):

        request = eval_data_item['question']

        summaries = []
        for batch_docs in batch_iterator(eval_data_item['doc_ctxs'], args.n_contexts):

            ## multi-doc summarization
            if 'prefix' in args.model_name_or_path:
                tokenized_inputs = collator
                tokenized_input = tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors='pt').to(model.device)

                outputs = generate_standard_with_prefix()
            else:
                outputs = generate_standard()


            summaries.extend(outputs)

        # add the new summaries 
        for i, summary in enumerate(summaries):
            eval_data_item['docs'][i][args.output_key] = summary

    if not os.path.exists("data/add_summary"):
        os.makedirs("data/add_summary")
    json.dump(eval_data, open(f"data/add_summary/{args.output_file}", "w"), indent=4)
