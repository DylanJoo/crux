import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import re
import argparse
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from glob import glob
import ir_measures
from ir_measures import RPrec, MAP
from transformers import AutoTokenizer

def load_qrel(path, threshold=1):
    data = defaultdict(list)
    if path is None:
        return None
    with open(path) as f:
        for line in f:
            item = line.strip().split()
            if int(item[3]) >= threshold:
                data[item[0]].append( item[2] )
    return data

def load_judgements(path, report_file=None):
    judgements = defaultdict(lambda: defaultdict(lambda: None))
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            example_id = data['example_id']
            judgements[example_id].update({data['pid']: data['rating']})

    if report_file is not None:
        with open(report_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                example_id = data['example_id']
                pid = f"{example_id}:report"
                judgements[example_id].update({pid: data['rating']})

    return judgements

def load_run(path, topk=9999):
    run = defaultdict(list)
    if path is None:
        return None
    with open(path, 'r') as f:
        for line in f:
            example_id, Q0, psgid, rank, score, prefix = line.strip().split()
            if int(rank) <= topk:
                run[example_id].append(psgid)
    return run

def load_passages(path):
    passages = {}
    if os.path.isdir(path):
        for file in glob(os.path.join(path, "*jsonl")):
            with open(file, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    passages[item["id"]] = item['contents']
    else:
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                passages[item["id"]] = item['contents']
    return passages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--generator_name", type=str, default=None)

    # base
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--rel_subset", type=int, default=3)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--weighted_factor", type=float, default=1)
    # context 
    parser.add_argument("--passage_path", type=str, default=None)
    parser.add_argument("--judgement_file", type=str, default=None)
    # ranking
    parser.add_argument("--run_file", type=str, default=None)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--report_file", type=str, default=None)
    args = parser.parse_args()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.generator_name)

    # load qrels, pre-calculated judge
    qrels = load_qrel(
        os.path.join(args.dataset_dir, f'ranking_{args.threshold}/{args.split}_qrels_oracle_context_pr.txt'),
        threshold=args.rel_subset
    )
    topic_ids = [q for q in qrels]

    runs = load_run(args.run_file, args.topk)

    # augmented context
    judgements_base = load_judgements(
        os.path.join(args.dataset_dir, f'ranking_{args.threshold}/{args.split}_judgements.jsonl')
    )
    passages_base = load_passages(
        os.path.join(args.dataset_dir, f'passages/{args.split}_psgs.jsonl')
    )

    # augmented context
    if 'vanilla' in args.tag:
        judgements = judgements_base
        passages = load_passages(args.passage_path)
    elif 'oracle-passage' in args.tag:
        judgements = judgements_base
        passages = passages_base
    else:
        judgements = load_judgements(args.judgement_file, report_file=args.report_file)
        passages = load_passages(args.passage_path)

    lengths = [len(p.split()) for p in passages.values()]
    dummy_passage = '0 ' * (sum(lengths) // len(lengths))
    n_questions = 10 if args.split == 'test' else 15

    # load graded passages
    outputs = {'coverage': [], 'density': [], 'num_segs': [], 'num_tokens': []}

    # oracle-report
    for example_id in qrels:

        # [oracle] inforamtion 
        ## upper bound of answerability --> cover
        psgids = [psgid for psgid in qrels[example_id]]
        judgement_oracle = np.array([judgements_base[example_id][psgid] for psgid in psgids]).max(0)
        answerable = judgement_oracle >= args.threshold

        ## upper bound of context length --> density 
        n_tokens_oracle = len(tokenizer.tokenize(
            " ".join([passages_base[psgid] for psgid in psgids])
        ))
        density_based = sum(answerable) / n_tokens_oracle

        # [retrieval-augmented] context information
        if ('oracle-report' in args.tag) or ('llmrg' in args.tag):
            psgids = [f'{example_id}:report']
        if runs is not None:
            psgids = runs[example_id]

        # collection prejudged ratings
        context = ""
        ratings = [[0] * n_questions]
        for psgid in psgids:

            ## answerbaility 
            if example_id == psgid.split(":")[0]: # only consider the context derieved from relevant
                judgement = judgements[example_id][psgid]
                ratings.append(judgement)

                if judgement is None:
                    logger.info(f"No judgement found.")

            ## context length 
            try:
                context = context + " " + passages[psgid]
            except:
                logger.info("No corresponding passages found for this setting.") 
                logger.info(f"You need to augment this passage {psgid} and add it to outputs.")
                logger.info(f"The evaluation results are computed by dummy passage for now. ")
                context = context + " " + dummy_passage 

        # get maximun ratings
        ratings = np.array(ratings).max(0)

        ## calculate coverage
        coverage = sum(ratings[answerable] >= args.threshold) / sum(answerable)
        outputs['coverage'].append(coverage)
        if ('oracle-report' in args.tag) and (coverage < 1):
            print(example_id)
            print(answerable)
            print(ratings[answerable])
            print(qrels[example_id])
            print([judgements_base[example_id][pid] for pid in qrels[example_id]])
            print(judgement_oracle[answerable])

        ## calculate density
        n_tokens = len(tokenizer.tokenize(context)) 
        density = sum(ratings[answerable] >= args.threshold) / n_tokens
        norm_density = (density / density_based) ** args.weighted_factor
        outputs['density'].append(norm_density)
        outputs['num_segs'].append(len(psgids))
        outputs['num_tokens'].append(n_tokens)

    # results
    mean_coverage = np.mean(outputs['coverage'])
    mean_density = np.mean(outputs['density'])
    mean_num_segments = np.mean(outputs['num_segs'])
    mean_num_tokens = np.mean(outputs['num_tokens'])
    num_coverage = len(outputs['coverage'])

    # results from ir_measures if have runs
    if runs is not None:
        qrels = ir_measures.read_trec_qrels(
            os.path.join(args.dataset_dir, f'ranking_{args.threshold}/{args.split}_qrels_oracle_context_pr.txt'),
        )
        runs = ir_measures.read_trec_run(args.run_file)
        rank_results = ir_measures.calc_aggregate([RPrec(rel=3), RPrec(rel=2), RPrec, MAP], qrels, runs)
        mean_rprec = (rank_results[RPrec(rel=3)], rank_results[RPrec(rel=2)], rank_results[RPrec])
        mean_ap = rank_results[MAP]
    else:
        mean_rprec = (1, 1, 1)
        mean_ap = 1

    # print results
    logger.info(f" # === Evaluation Results === ")
    logger.info(f" # TAG : {args.tag} | {num_coverage} examples")
    logger.info(f' # Mean Coverage     (tau={args.threshold}) : {mean_coverage:.4f}')
    logger.info(f' # Mean Norm-Density (tau={args.threshold}) : {mean_density:.4f}')
    logger.info(f' # RPrec (mu=3/2/1)  (tau={args.threshold}) : {mean_rprec[0]:.4f}, {mean_rprec[1]:.4f}, {mean_rprec[2]:.4f}')
    logger.info(f' # MAP @ {args.topk}          (tau={args.threshold}) : {mean_ap:.4f}')
    logger.info(f' # Mean number of segments     : {mean_num_segments:.2f}')
    logger.info(f' # Mean number of tokens       : {mean_num_tokens:.2f}\n')

    print(f"  {args.tag} | {mean_num_segments:.2f} | {mean_num_tokens:.2f} |" + \
          f" {mean_coverage:.4f} | {mean_density:.4f} |" + \
          f" {mean_rprec[0]:.4f} - {mean_rprec[1]:.4f} - {mean_rprec[2]:.4f} | {mean_ap:.4f}")
