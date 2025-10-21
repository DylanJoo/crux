# TODO: add density measure
# TODO: add t-test with run_b
# NOTE: integrate to ir_measures ?

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import sys
import os
import json
import argparse
import numpy as np
from crux.tools import load_run_or_qrel, load_diversity_qrel, load_ratings
from collections import defaultdict
import ir_measures
from ir_measures import Metric, MAP, nDCG, P, alpha_nDCG
import pdb

def coverage_meausres(ratings, ratings_oracle, filter_by_oracle=False, tau=3):
    # get answerable amount
    if filter_by_oracle:
        answerable = np.bool([r>=tau for r in ratings_oracle])
    else:
        answerable = np.bool([1 for r in ratings])
    value = sum(ratings[answerable] >= tau) / sum(answerable)
    metric = Metric(query_id='dummy', value=value, measure='Cov')
    return metric

def rac_eval(run, qrel, div_qrel, judge, tau=3, cutoff=10, filter_by_oracle=False, run_b=None):
    outputs = defaultdict(list)

    for metric in ir_measures.iter_calc([nDCG@10, P@10], qrel, run):
        outputs[metric.measure.NAME + "@10"].append(metric.value)

    for metric in ir_measures.iter_calc([alpha_nDCG@10], div_qrel, run):
        outputs[metric.measure.NAME + "@10"].append(metric.value)

    # NOTE: Density measure TBD.
    for qid in run: 
        ### NOTE: the retrieval context aggregation is `max` however it can be changed to others
        ratings_oracle = np.max([judge[qid][docid] for docid in judge[qid]], 0)
        empty = [0] * len(ratings_oracle)
        ratings = np.max([judge[qid][docid] if docid in judge[qid] else empty for docid in run[qid]][:cutoff], 0)
        assert len(ratings) == len(ratings_oracle), \
                f"Inconsistent ratings length, got {len(ratings)} vs {len(ratings_oracle)}"

        metric = coverage_meausres(
            ratings=ratings, 
            ratings_oracle=ratings_oracle, 
            filter_by_oracle=filter_by_oracle,
            tau=tau
        )
        outputs[metric.measure +"@10"].append(float(metric.value))

    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Path to the run file")
    parser.add_argument("--qrel", type=str, required=True, help="Path to the qrel file")
    parser.add_argument("--judge", type=str, required=True, help=\
            "jsonl: {'id': str, 'docid': List[str], 'ratings': List[int]'}")
    parser.add_argument("--run_b", type=str, default=None)
    parser.add_argument("--filter_by_oracle", action="store_true", default=False)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    args = parser.parse_args()

    # load data
    run = load_run_or_qrel(args.run, topk=10)
    qrel = load_run_or_qrel(args.qrel, threshold=1) # NOTE: only support binary labels so far
    div_qrel = load_diversity_qrel(args.qrel) # NOTE: only support binary labels so far
    runb = load_run_or_qrel(args.run_b, topk=10) if args.run_b is not None else None
    ratings = load_ratings(args.judge)

    ## sanity check
    missing_qids = [qid for qid in run.keys() if qid not in qrel]
    if len(missing_qids) > 0:
        logger.warning(f"Missing results: {len(missing_qids)} / {len(run)} -> {missing_qids}.")

    # run eval
    outputs = rac_eval(
        run=run, 
        qrel=qrel, div_qrel=div_qrel, 
        run_b=runb, 
        tau=3,
        cutoff=10,
        judge=ratings, 
        filter_by_oracle=args.filter_by_oracle, 
    )

    # Get evearge across queries
    sys.stdout.write(args.run.rsplit('/', 1)[0] + " | " + args.run.rsplit('/')[-1] + " | ")
    for key, values in outputs.items():
        sys.stdout.write(key + " | ")
        sys.stdout.write("{:.4f}".format(np.mean(values)) + " | ")
    sys.stdout.write("\n")

