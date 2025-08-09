import os
import json
import argparse
from tqdm import tqdm
from crux.tools import batch_iterator
from crux.tools.researchy.ir_utils import load_topic, load_queries

def main(args):

    # Data 
    if args.query_type == "init-q":
        queries = load_topic(hf_dataset_name="corbyrosset/researchy_questions")

    if args.query_type == "gpt4-q":
        queries = load_queries(hf_dataset_name="corbyrosset/researchy_questions")

    if args.data.num_shards > 0:
        shard_size = len(queries) // args.data.num_shards
        qids = list(queries.keys())
        qids = qids[args.data.shard * shard_size: (args.data.shard + 1) * shard_size]
        queries = {qid: queries[qid] for qid in qids}

    if len(queries) == 0:
        print("No queries to process. Exiting.")
        return

    # Retrieval
    from crux.sim_rag.retrieve.bm25 import search
    output_run = search(
        index=args.data.index_dir,
        k1=args.retrieval.k1, 
        b=args.retrieval.b,
        topics=queries,
        batch_size=args.retrieval.batch_size,
        k=args.retrieval.k,
        writer=open(args.data.output_run + f".shard{args.data.shard}" if args.data.num_shards > 1 else "", "w"),
        stemming=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run researchy queries with BM25 retrieval.")
    parser.add_argument("--shard", type=int, default=0, help="Shard number for distributed processing.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards for distributed processing.")
    cli_args = parser.parse_args()

    from types import SimpleNamespace
    for q_type in ["gpt4-q"]:
        args = SimpleNamespace(
            data=SimpleNamespace(
                index_dir="/exp/ayates/clueweb22-index/en-all_porter_stemmer/",
                output_run=f"/exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-{q_type}_bm25-clueweb22-b.txt",
                shard=int(cli_args.shard),
                num_shards=int(cli_args.num_shards)
            ),
            query_type=q_type,
            retrieval=SimpleNamespace(
                k1=0.9,
                b=0.4,
                batch_size=64,
                k=100
            )
        )
        main(args)
