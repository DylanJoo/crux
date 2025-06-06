import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re
import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from glob import glob
from utils import replace_tags, load_topics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", type=str, default=None)
    parser.add_argument("--ratings_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--random_subset", type=int, default=-1)
    args = parser.parse_args()

    ## topic subset
    topics_all = []
    logger.info("load topics ...") 
    for file in tqdm(glob(os.path.join(args.shard_dir, f"topics-gen/*-{args.split}-*.json"))):
        topic = load_topics(file)
        topics_all += topic
    if args.random_subset > 0:
        np.random.seed(args.random_subset)
        selected = np.random.randint(0, len(topics_all), args.random_subset)
        topics_all = {r['example_id']: r['texts'] for i, r in enumerate(topics_all) if i in selected}
    else:
        topics_all = {r['example_id']: r['texts'] for r in topics_all}

    # Load context
    files = glob(os.path.join(args.ratings_dir, f"*-{args.split}-*"))

    passages = {}
    documents = {}
    for file in files:
        with open(file, 'r') as f:
            for line in tqdm(f, "load contexts"):
                data = json.loads(line.strip())
                example_id = data['example_id']
                documents_ = data['documents']
                passages_ = data['passages']
                n_passages = sum(len(plist) for plist in passages_)
                questions, ratings = data['questions'], data['ratings']
                ratings = np.array(ratings)

                # sanity check
                if example_id not in topics_all:
                    continue 

                if ratings.shape != (n_passages, len(questions)):
                    logging.warnings(f"example id: {example_id} has incorrect number of passages: {n_passages} ({len(questions)} questions).")
                    continue
                if len(documents_) != len(passages_):
                    logging.warnings(f"example id: {example_id} has inconsistent number of context: #d: {len(documents_)} #p: {len(passages_)}.")
                    continue

                j = 0
                for i, plist in enumerate(passages_):
                    docid = f"{example_id}:{i}"
                    documents[docid] = documents_[i]

                    for passage in plist:
                        psg_id = f"{docid}#{j}"
                        passages[psg_id] = passage
                        j += 1

    # Save the context
    output_dir = os.path.join(args.output_dir, 'documents')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.split}_docs.jsonl")
    with open(output_file, 'w') as f:
        for docid, document in documents.items():
            f.write(json.dumps({"id": docid, "contents": document}, ensure_ascii=False)+'\n')

    output_dir = os.path.join(args.output_dir, f'passages')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.split}_psgs.jsonl")
    with open(output_file, 'w') as f:
        for psgid, passage in passages.items():
            f.write(json.dumps({"id": psgid, "contents": passage}, ensure_ascii=False)+'\n')

    print("done")
