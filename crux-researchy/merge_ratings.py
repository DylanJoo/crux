import re
import os
import json
import argparse
import glob
from tqdm import tqdm
from crux.tools import batch_iterator, load_ratings
from crux.tools.researchy.ir_utils import load_topic, load_subtopics

def load_offload_jsonl(ratings, offload_dir):
    subtopics = load_subtopics()

    files = glob.glob(os.path.join(offload_dir, "*.jsonl"))

    for file in files:
        with open(file, 'r') as f:
            for line in f:
                item = json.loads(line)
                qid, docid, i_subquestion = item['id'].split("::")
                llm_output = item['response']['body']['choices'][0]['message']['content'] 

                ## init qid
                if qid not in ratings:
                    ratings[qid] = {}

                if docid not in ratings[qid]:
                    ratings[qid][docid] = [None for _ in range(len(subtopics[qid]))]

                ## postprocess 
                pattern = re.compile(r"\d|-\d")
                rating = re.findall(pattern, llm_output + " -1")
                rating = int(rating[0])
                ratings[qid][docid][int(i_subquestion)] = rating

    # sanity check
    return ratings

def main(args):
    # Get input and outputs
    # /exp/ayates/scale25/batch-vllm/output/ratings.Llama-3.3-70B-Instruct.*qwen*
    input_ratings=f"/exp/scale25/artifacts/crux/crux-researchy/judge/ratings.Llama-3.3-70B-Instruct.{args.shard}-{args.num_shards}.jsonl"
    # input_ratings=f"/exp/scale25/artifacts/crux/crux-researchy/judge/ratings.Llama-3.3-70B-Instruct.{args.shard}-{args.num_shards}.jsonl"
    offload_dir=f"/exp/ayates/scale25/batch-vllm/output/ratings.Llama-3.3-70B-Instruct.{args.shard}-{args.num_shards}.qwen0.6b-top10"
    output_path=f"/exp/scale25/artifacts/crux/crux-researchy/judge-offload/ratings.Llama-3.3-70B-Instruct.{args.shard}-{args.num_shards}.jsonl"

    # Data 
    queries = load_topic()

    if args.num_shards > 0:
        shard_size = len(queries) // args.num_shards
        qids = list(queries.keys())
        qids = qids[args.shard * shard_size: (args.shard + 1) * shard_size]
        queries = {qid: queries[qid] for qid in qids}

    if len(queries) == 0:
        print("No queries to process. Exiting.")
        return

    # Load judged ratings
    if os.path.exists(input_ratings):
        ratings = load_ratings(input_ratings)
        print(f"{args.shard} - {args.num_shards} - Prejudged ratings for {len(ratings)} queries loaded.") 
    else:
        ratings = {}

    if os.path.exists(offload_dir):
        ratings = load_offload_jsonl(ratings, offload_dir)

    print(f"{args.shard} - {args.num_shards} - Total ratings for {len(ratings)} queries")

    # write them out
    with open(output_path, 'w') as f:
        for qid in ratings:
            # check if all subquestions have a rating
            for docid in ratings[qid]:
                if all([r != None for r in ratings[qid][docid]]):
                    rating = ratings[qid][docid]
                    f.write(json.dumps({"id": qid, "docid": docid, "rating": rating }) + "\n")
                else:
                    print(f"Incomplete ratings for {qid} - {docid}: {ratings[qid][docid]}")

    print(f"Total items saved: {len(ratings)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run researchy queries with BM25 retrieval.")
    parser.add_argument("--shard", type=int, default=0, help="Shard number for distributed processing.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards for distributed processing.")
    cli_args = parser.parse_args()

    from types import SimpleNamespace
    args = SimpleNamespace(shard=int(cli_args.shard), num_shards=int(cli_args.num_shards))
    main(args)
