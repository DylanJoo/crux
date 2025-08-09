import os
import json
import argparse
from tqdm import tqdm
from crux.tools import batch_iterator, load_run_or_qrel
from crux.tools.researchy.ir_utils import create_subset_corpus, get_qrel

def main(args):

    # Data  (more advace pooling?)
    run = load_run_or_qrel(args.data.input_run, topk=100, threshold=3)
    qrel = get_qrel() # make this as qrel? NOTE: this can be eval metric for pool quality
    pooled_docs_run = [doc_id for id_scores in run.values() for doc_id in id_scores]
    pooled_docs_qrel = [doc_id for id_scores in qrel.values() for doc_id in id_scores]
    pooled_docs = set(pooled_docs_run + pooled_docs_qrel)

    # Retrieval
    corpus = create_subset_corpus(
        glob_path="/export/common/data/corpora/clueweb22-b/txt/en/en00/en00*/en*json.gz",
        subset=pooled_docs,
        shard=args.data.shard,
        num_shards=args.data.num_shards,
    )

    # Save corpus
    with open(args.data.output_corpus + f".shard{args.data.shard}" if args.data.num_shards > 1 else "", "w") as f:
        for docid, doc in corpus.items():
            f.write(json.dumps(doc) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run researchy queries with BM25 retrieval.")
    parser.add_argument("--shard", type=int, default=0, help="Shard number for distributed processing.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards for distributed processing.")
    cli_args = parser.parse_args()

    from types import SimpleNamespace
    args = SimpleNamespace(
        data=SimpleNamespace(
            index_dir="/exp/ayates/clueweb22-index/en-all/",
            input_run="/exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-init-q.bm25.clueweb22-b.txt",
            output_corpus="/exp/scale25/artifacts/crux/crux-researchy/docs/cw22-b.researchy-v1/doc00.jsonl",
            shard=int(cli_args.shard),
            num_shards=int(cli_args.num_shards)
        ),
    )

    os.makedirs(os.path.dirname(args.data.output_corpus), exist_ok=True)
    main(args)
