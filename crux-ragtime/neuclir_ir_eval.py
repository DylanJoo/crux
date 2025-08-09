import os
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import asyncio
import aiohttp
import argparse
import json
import pprint

import ir_measures
from ir_measures import nDCG
from tqdm.asyncio import tqdm

from tools.neuclir.ir_utils import load_qrels, load_topic
from tools.search.search import retrieve_with_subqueries

async def search_neuclir(args, session, query, search_endpoint, service_name, limit, 
                         subset=None, max_limit=500, **kwargs):
    config = {"service": service_name, "query": str(query), "limit": limit, **kwargs}
    if subset != "mlir":
        config.update({"subset": subset})

    if args.search_with_subqueries:
        _config = {k: v for k, v in config.items() if k not in ["service", "query", "limit"]}
        _config["session"] = session
        args.host = args.service_endpoint
        result = await retrieve_with_subqueries(args, query, service_name, 
                                                return_doc_scores=True, **_config)
        if len(result) >= 20:
            return result, 1
        else:
            logger.warning(f"Retrieved less than 20 docs: {len(result)}")
    else:
        # set higher timeout as reranking takes times
        for l in range(limit, max_limit+100, 100):
            config['limit'] = min(l, max_limit)
            async with session.post(search_endpoint + "/query", json=config) as response:
                result = await response.json()
                result = {k: v for k, v in sorted(result["result"].items(), key=lambda item: item[1], reverse=True)}

                if len(result) >= 20:
                    return result, 1

    return result, 0

# [TODO] Add local evalaution with (1) local run file or (2) local topic file
async def ir_eval(args):

    ir_results = {}
    for year in ["2022", "2023", "2024"]:
        ir_results[year] = {}
        # for lang in ["zho", "rus", "fas", "mlir"]:
        for lang in ["mlir"]:
            if year == "2022" and lang == "mlir":  # exclude 2022
                continue
            else:
                qrels = load_qrels(f"/expscratch/eyang/collections/neuclir/qrels/{year}.{lang}.qrels")
                topic_path = f"/expscratch/eyang/collections/neuclir/topics/{year}.jsonl" or args.topic_path
                queries = load_topic(topic_path)
                queries = {k: v for k, v in queries.items() if k in qrels.keys()}

                config = {
                    'limit': args.limit, 
                    'max_limit': (args.max_limit or args.limit),
                    'subset': lang,
                    'search_endpoint': args.service_endpoint if 'http' in args.service_endpoint else f"http://{args.service_endpoint}",
                    'service_name': args.service_name,
                }

                qids, query_txts = zip(*queries.items())

                timeout = aiohttp.ClientTimeout(total=3600, connect=None, sock_connect=None, sock_read=None)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    results_flags = await tqdm.gather(
                        *[search_neuclir(args, session=session, query=query, **config) for query in query_txts],
                        total=len(qids),
                        desc=f"Searching NeuCLIR {year} {lang}",
                    )

                results, flags = zip(*results_flags)
                failed = [qid for qid, flag in zip(qids, flags) if flag == 0] # failed qids
                runs = dict(zip(qids, results))

                score = ir_measures.calc_aggregate([nDCG@20], qrels, runs)[nDCG@20]
                logger.info(f"Year: {year}, Language: {lang}, nDCG@20: {score:.4f}, # Failed: {len(failed)}/{len(qids)}")
                ir_results[year][lang] = score

                # save run file
                run_path = os.path.join(args.output_dir, f"neuclir-{year}-{lang}.{args.service_name}.run")
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
    parser.add_argument("--max_limit", type=int, default=None)
    parser.add_argument("--run_path", type=str, default=None, help="The path to the retrieval result file (top100).")
    parser.add_argument("--topic_path", type=str, default=None, help="Use customized query")
    parser.add_argument("--output_dir", type=str, default="results.search_service", help="Dir of results.")
    parser.add_argument("--model", type=str, default="llama3.3-70b-instruct", help="Model to use")
    parser.add_argument("--search_with_subqueries", default=False, action="store_true", help="When set to True, search will be performed using query decomposition")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"IR Evaluation -- {args.service_name}")
    ir_results = asyncio.run(ir_eval(args))
    pprint.pprint(ir_results)

    with open(os.path.join(args.output_dir, 'ir_results.json'), "w") as f:
        f.write(json.dumps(ir_results, indent=4))
    logger.info(f"Results saved to {args.output_dir}/*")
