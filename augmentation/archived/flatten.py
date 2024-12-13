import os
import re
import argparse
import json
from tqdm import tqdm
from glob import glob
from utils import load_passages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True) 

    # passages extracted from docs
    passages_all = []
    files = glob(os.path.join(args.input_dir, "*summ*.json"))
    for file in tqdm(files):
        passages = load_passages(file)
        passages_all += passages
    passages_all = {p['example_id']: p['texts'] for p in passages_all}

    with open(os.path.join(args.output_dir, "doc_to_passages.jsonl"), 'w') as f:
         for example_id in tqdm(passages_all, total=len(passages_all)):
             j = 0
             for i, passages in enumerate(passages_all[example_id]):
                 for passage in passages:
                     f.write(json.dumps({
                         "id": f"{example_id}#{i}:{j}", 
                         "contents": passage
                     })+'\n')
                     j += 1

