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

def load_topic(fields):
    data = json.load(open('/exp/scale25/biogen/starter-kit-2025/data/task_b.json', 'rb'))
    topic = {}
    if isinstance(fields, str):
        fields = [fields]
    for item in data:
        text = " ".join([item[field] for field in fields])
        topic[item['id']] = text
    return topic

async def search(args, session, query, search_endpoint, service_name, limit, **kwargs):
    config = {"service": service_name, "query": str(query), "limit": limit, **kwargs}

    # set higher timeout as reranking takes times
    async with session.post(search_endpoint + "/query", json=config) as response:
        result = await response.json()
        result = {k: v for k, v in sorted(result["result"].items(), key=lambda item: item[1], reverse=True)}

        if len(result) >= 20:
            return result, 1

    return result, 0

async def ir_eval(args):

    ir_results = {}
    topic = load_topic(args.fields)

    config = {
        'limit': args.limit, 
        'search_endpoint': args.service_endpoint if 'http' in args.service_endpoint else f"http://{args.service_endpoint}",
        'service_name': args.service_name,
    }

    qids, query_txts = zip(*topic.items())
    print(query_txts[:5])

    async with aiohttp.ClientSession() as session:
        results_flags = await tqdm.gather(
            *[search(args, session=session, query=query, **config) for query in query_txts],
            total=len(qids),
            desc=f"Searching: {args.service_name}",
        )

    results, flags = zip(*results_flags)
    failed = [qid for qid, flag in zip(qids, flags) if flag == 0] # failed qids
    runs = dict(zip(qids, results))

    # save run file
    prefix = "+".join(args.fields) if isinstance(args.fields, list) else args.fields
    run_path = os.path.join(args.output_dir, f"{prefix}_{args.service_name}.run")
    with open(run_path, 'w') as f:
        for qid, docs in runs.items():
            for rank, (docid, score) in enumerate(docs.items(), start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score} {args.service_name}\n")

    return ir_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--service_name", type=str, default="plaidx-neuclir", help="The search service online.")
    parser.add_argument("--service_endpoint", type=str, default="10.162.95.158:5000")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--topic_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fields", action='append', default=[])
    args = parser.parse_args()
    print(args)

    os.makedirs(args.output_dir, exist_ok=True)
    asyncio.run(ir_eval(args))

