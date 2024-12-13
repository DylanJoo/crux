import os
import torch
import json
import argparse
from tqdm import tqdm
from utils import batch_iterator, load_model

def truncate_and_concat(texts, tokenizer, max_length=512):
    tokenized = tokenizer.tokenize(texts)
    length = len(tokenizer.tokenize(texts))
    max_length = (max_length or tokenizer.max_lengt_single_sentence-1)
    if (length+6) < max_length:
        return text
    else:
        return tokenizer.convert_tokens_to_string(tokenized[:(max_length-6)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--model_class", type=str, default=None, choices=["fid", "seq2seq", "causualLM"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_key", type=str, default="summary_debug")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--truncate", type=bool, default=False, action='store_true')
    parser.add_argument("--template", type=str, default="title: {T} content: {P}")
    args = parser.parse_args()

    # load model
    model, tokenizer = load_model(args.model_name_or_path, model_class=args.model_class)
    model.eval()

    # load writer and evaluation data 
    eval_data = json.load(open(args.eval_file))

    for eval_data_item in tqdm(eval_data, total=len(eval_data)):

        # multiple retrieved document here
        request = eval_data_item['question']
        request_context = eval_data_item['question_ctx']

        # batch inference
        summaries = []
        for batch_docs in batch_iterator(eval_data_item['docs'], args.batch_size):

            if truncate:
                batch_docs = [truncate_and_concat(doc) for doc in batch_docs]

            input = list(
                args.template.replace("{Q}", request).replace("{T}", doc['title']).replace("{P}", doc['text']) \
                        for doc in batch_docs
            )
            tokenized_input = tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors='pt').to(model.device)
            outputs = model.generate(**tokenized_input, max_new_tokens=100)

            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend(outputs)

        # add the new summaries 
        for i, summary in enumerate(summaries):
            eval_data_item['docs'][i][args.output_key] = summary

    if not os.path.exists("data/add_summary"):
        os.makedirs("data/add_summary")
    json.dump(eval_data, open(f"data/add_summary/{args.output_file}", "w"), indent=4)

if __name__ == '__main__':
    with torch.no_grad():
        main()

