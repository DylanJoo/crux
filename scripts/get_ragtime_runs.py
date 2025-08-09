"""usage:
topic_file=/exp/scale25/ragtime/topics/topic25_dry.jsonl
service=qwen3-listllama-neuclir

python get_ragtime_runs.py \
    --service_endpoint http://rack2n02:5000 \
    --service_name ${service} \
    --topics_path ${topic_file} \
    --output_dir neuclir_runs \
    --prefix neuclir_ps \
    --limit 1000
"""
import os
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import asyncio
import aiohttp
import argparse
from argparse import Namespace
import json
import pprint

import ir_measures
from ir_measures import nDCG
from tqdm.asyncio import tqdm

# from tools.search.search import retrieve_with_subqueries

def load_query(args: Namespace):
    """Returns:
    queries [dict]: queries used as part of CRUX
    queries_for_search [dict]: queries used when performing search
                               (i.e. we might not want to include the background)
    raw_topics [dict]: raw topics as provided in topic file (i.e. example-request.jsonl)
    """
    path = args.topics_path
    queries = {}
    queries_for_search = {}
    raw_topics = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            queries[data["request_id"]] = data["problem_statement"].strip()
            queries_for_search[data["request_id"]] = data["title"].strip() + " " + data["problem_statement"].strip()  # don't add background
            raw_topics.append(data)
    return queries, queries_for_search

async def search_neuclir(args, session, query, search_endpoint, service_name, limit, subset=None, **kwargs):
    config = {"service": service_name, "query": str(query), "limit": limit, **kwargs}

    # set higher timeout as reranking takes times
    async with session.post(search_endpoint + "/query", json=config) as response:
        result = await response.json()
        result = {k: v for k, v in sorted(result["result"].items(), key=lambda item: item[1], reverse=True)}

        if len(result) >= 20:
            return result, 1

    return result, 0

# [TODO] Add local evalaution with (1) local run file or (2) local topic file
async def ir_eval(args):

    ir_results = {}
    queries_types = load_query(args)
    if 'ps' in args.prefix:
        queries = queries_types[0]
    if 't+ps' in args.prefix:
        queries = queries_types[1]

    queries = {k: v for k, v in queries.items()}

    config = {
        'limit': args.limit, 
        'search_endpoint': args.service_endpoint if 'http' in args.service_endpoint else f"http://{args.service_endpoint}",
        'service_name': args.service_name,
    }

    qids, query_txts = zip(*queries.items())

    timeout = aiohttp.ClientTimeout(total=7200, connect=None, sock_connect=None, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results_flags = await tqdm.gather(
            *[search_neuclir(args, session=session, query=query, **config) for query in query_txts],
            total=len(qids),
            desc=f"Searching for requests in {args.topics_path}",
        )

    results, flags = zip(*results_flags)
    failed = [qid for qid, flag in zip(qids, flags) if flag == 0] # failed qids
    runs = dict(zip(qids, results))

    # save run file
    run_path = os.path.join(args.output_dir, f"{args.prefix}_{args.service_name}.run")
    with open(run_path, 'w') as f:
        for qid, docs in runs.items():
            for rank, (docid, score) in enumerate(docs.items(), start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score} {args.service_name}\n")

    return ir_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--service_name", type=str, default="plaidx-neuclir", help="The search service online.")
    parser.add_argument("--service_endpoint", type=str, default="10.162.95.158:5000")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--topics_path", type=str, default=None, help="Use customized query")
    parser.add_argument("--output_dir", type=str, default=None, help="Use customized query")
    parser.add_argument("--prefix", type=str, default=None, help="dataset and query type")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    asyncio.run(ir_eval(args))
