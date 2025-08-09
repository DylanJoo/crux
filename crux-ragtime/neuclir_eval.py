import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import os
import json
import requests
import argparse
import re
from tqdm import tqdm

import pprint
import ir_measures
from ir_measures import R, nDCG

from tools.neuclir.ir_utils import load_qrels, load_topic

def search_neuclir(
    query, 
    search_endpoint,
    service_name,
    limit=100, 
    subset=None, 
    **kwargs
):
    data = {
        'service': service_name,
        'query': str(query),
        'limit': limit,
        **kwargs
    }
    if subset != 'mlir':
        data.update({"subset": subset})

    result = requests.post(search_endpoint + "/query", json=data).json()
    result = {k: v for k, v in sorted(result['result'].items(), key=lambda item: item[1], reverse=True)}
    return result

# [TODO] Integrate both of IR/RAC evaluation into a single script
# [TODO] Add local evalaution with run_path
def ir_eval(args):
    ir_results = {}
    # for year in ['2022', '2023', '2024']: 
    for year in ['2022']: 
        ir_results[year] = {}
        # for lang in ['zho', 'rus', 'fas', 'mlir']:
        for lang in ['fas']:
            if (year=='2022' and lang=='mlir'): # exclude 2022
                continue
            else:
                qrels = load_qrels(f'/expscratch/eyang/collections/neuclir/qrels/{year}.{lang}.qrels')
                topic_path = f'/expscratch/eyang/collections/neuclir/topics/{year}.jsonl' or args.topic_path
                queries = load_topic(topic_path)
                queries = {k: v for k, v in queries.items() if k in qrels.keys()}

                runs = {}
                config = {
                    'limit': args.topk, 'subset': lang,
                    'search_endpoint': f"http://{args.service_endpoint}", 
                    'service_name': args.service_name,
                }
                for qid, query in tqdm(queries.items(), desc=f"Searching NeuCLIR {year} {lang}", total=len(queries)):
                    runs[qid] = search_neuclir(
                        query=query, 
                        **config
                    )

                score = ir_measures.calc_aggregate([nDCG@20], qrels, runs)[nDCG@20]
                logger.info(f"Year: {year}, Language: {lang}, nDCG@20: {score:.4f}")
                ir_results[year][lang] = score 

    pprint.pprint(ir_results)

    return ir_results

def rac_eval(args):
    rac_results = {}
    pprint.pprint(rac_results)
    return rac_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--service_name", type=str, default="plaidx-neuclir", help="The search service online.")
    parser.add_argument("--service_endpoint", type=str, default="10.162.95.158:5000")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--run_path", type=str, default=None, help="The path to the retrieval result file (top100).")
    parser.add_argument("--topic_path", type=str, default=None, help="Use customized query")
    parser.add_argument("--output_path", type=str, default=None, help="Output result file path.")
    args, _ = parser.parse_known_args()

    all_results = {'ir_results': {}, 'rac_results': {}}

    logger.info(f'IR Evaluation -- {args.service_name}')
    ir_results = ir_eval(args)
    all_results['ir_results'] = ir_results

    logger.info(f'RAC Evaluation -- {args.service_name}')
    rac_results = rac_eval(args)
    all_results['rac_results'] = rac_results

    output_file = args.output_path or 'neuclir_eval.json'
    with open(output_file, 'w') as f:
        f.write(json.dumps(ir_results, indent=4))
