import json
import os
import torch
from tqdm import tqdm
import argparse
from collections import defaultdict
import sys
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from glob import glob

def batch_iterator(
    iterable, 
    size=1, 
    return_index=False
):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def sort_dict(
    dictionary, 
    quantization_factor,
    minimum
):
    d = {k: v*quantization_factor for (k, v) in dictionary if v >= minimum}
    sorted_d = {reverse_voc[k]: round(v, 3) for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    return sorted_d

def generate_vocab_vector(
    docs, 
    model, 
    minimum=0, 
    device='cpu', 
    max_length=512, 
    quantization_factor=100
):
    # now compute the document representation
    inputs = tokenizer(
        docs, 
        return_tensors="pt", 
        padding='max_length', 
        truncation=True, 
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        model_output = model(**inputs)
        logits = model_output.logits 
        doc_reps, _ = torch.max(
            torch.log(1 + torch.relu(logits)) 
            * inputs['attention_mask'].unsqueeze(-1), dim=1
        )

    # get the number of non-zero dimensions in the rep:
    cols = torch.nonzero(doc_reps)

    # now let's inspect the bow representation:
    weights = defaultdict(list)
    for col in cols:
        i, j = col.tolist()
        weights[i].append( (j, doc_reps[i, j].cpu().tolist()) )

    return [sort_dict(weight, quantization_factor, minimum) for i, weight in weights.items()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--collection_dir", type=str)
    parser.add_argument("--collection_output", type=str)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--quantization_factor", type=int, default=1000)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--minimum", type=float, default=0)
    args = parser.parse_args()

    # load models
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path or args.tokenizer_name)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

    # load data
    collection = []
    for collection_file in glob(os.path.join(args.collection_dir, "*jsonl")):
        with open(collection_file, 'r') as f:
            for line in tqdm(f):
                item = json.loads(line.strip())
                collection.append(item)
    dataset = Dataset.from_list(collection)
    del collection 
    print(dataset)

    # preparing batch 
    vectors = []
    data_iterator = batch_iterator(dataset, args.batch_size, False)

    output_dir = os.path.dirname(args.collection_output)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.collection_output, 'w') as fout:
        for batch in tqdm(data_iterator, total=len(dataset)//args.batch_size+1):
            batch_vectors = generate_vocab_vector(
                    docs=batch['contents'], 
                    model=model,
                    minimum=args.minimum,
                    device=args.device,
                    max_length=args.max_length,
                    quantization_factor=args.quantization_factor
            )
            vectors += batch_vectors

            # collection and re-dump the collections
            n = len(batch['id'])
            for i in range(n):
                example = {
                    "id": batch['id'][i],
                    "contents": batch['contents'][i],
                    "title": batch['title'][i] if 'title' in batch else "",
                    "vector": batch_vectors[i]
                }
                fout.write(json.dumps(example, ensure_ascii=False)+'\n')
