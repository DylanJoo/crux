import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from collections import defaultdict
import os
import re
import argparse
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import ir_measures
from ir_measures import RPrec, R, MAP, nDCG, alpha_nDCG, ScoredDoc
from transformers import AutoTokenizer

from ...tools.neuclir.ir_utils import load_diversity_qrels


def get_runs_from_rac_data(rac_data: defaultdict) -> list:
    """ Get runs from 'rac_data'. As in the reference qrel file,
        a document is assigned relevance=3 if the document is present in the
        retrieved documents. """
    runs = [
        ScoredDoc(qid, docid, 3)
        for qid, doc_scores in rac_data.items()
        for docid in doc_scores["docids"]
    ]

    return runs

def rac_evaluate(
    args, qrels, judgements,
    rac_data,
    diversity_qrels_path=None,
    threshold=3,     # answerability threshold (tau)
    rel_threshold=3, # on qrel's last column
    runs=None,
    tokenizer_name=None,
    gamma=0.5, tag='experiment',
    aggregation='max',
):
    
    diversity_qrels = load_diversity_qrels(diversity_qrels_path)

    outputs = {'coverage': [], 'coverage_at_10': [], 'density': [], 'num_segs': [], 'num_tokens': []}

    overlapped = {k: v for k, v in qrels.items() if k in rac_data}
    #overlapped_diversity = [q for q in diversity_qrels if q.query_id in rac_data]
    max_k = {}

    if len(overlapped) != len(qrels):
        logger.warning(' #Topics in qrels and rac_data are not consistent.' + \
                f' Got {len(qrels)} and {len(rac_data)}.')
        qrels = overlapped
        #diversity_qrels = overlapped_diversity

    for qid in tqdm(qrels, desc='RAC Evaluating', total=len(qrels)):

        """
        # [oracle] 
        docids = [docid for docid, score in qrels[qid].items() if score >= rel_threshold ] 
        #rac_text = " ".join([corpus[docid]['text'] for docid in docids])
        #n_tokens = len(tokenizer.tokenize(rac_text))

        #judgement_oracle = np.array([judgements[qid][docid] for docid in docids]).max(0)
        n_questions = len(next(iter(judgements[qid].items()))[1])
        judgements_oracle = [judgements[qid].get(docid, [3] * n_questions) for docid in docids] 
                                                 # TODO: qrels_obj.get_nuggets_answered_by_doc_id(qid, docid, top_k=10)) for docid in docids]
        judgement_oracle = np.array(judgements_oracle).max(0)
        answerable = (judgement_oracle >= threshold)
        #density_oracle = sum(answerable) / n_tokens
        """
        
        n_questions = len(next(iter(judgements[qid].items()))[1])
        answerable = np.full((n_questions,), np.True_, dtype=bool) # when using human nugget references

        # [retrieval-augmented context] 
        rac_type = rac_data[qid]['type']
        rac_text = " ".join(rac_data[qid]['context_list'])
        docids = rac_data[qid]['docids']
        #max_k[qid] = rac_type[1] # align to the max_k used in rac

        if 'oracle-report' in tag:
            docids = [f'{qid}:report']

        
        ## answerability
        ratings = []
        for docid in docids:
            if qid == docid.split(":")[0]: # only consider the context derieved from relevant
                if not ratings:
                    ratings = [[0] * n_questions]
            judgement = judgements[qid][docid]
            if not judgement: # if judgement cannot be found, assign 0 for each subquestion
                #logger.warning(f"qid {qid}, pid {docid} is not found in judgements, assigning 0 to all CRUX sub-questions")
                judgement = [0] * n_questions
            ratings.append(judgement)
        top_10_ratings = ratings[:10]

        if aggregation == 'max':
            ratings = np.array(ratings).max(0)
            top_10_ratings = np.array(top_10_ratings).max(0)
        elif aggregation == 'mean':
            ratings = np.array(ratings).mean(0)
        elif aggregation == 'mean_over_count':
            ratings = np.array(ratings).sum(0) / (np.array(ratings) != 0).sum(0)

        # [calculate] coverage
        coverage = sum(ratings[answerable] >= threshold) / sum(answerable)
        coverage_at_10 = sum(top_10_ratings[answerable] >= threshold) / sum(answerable)

        # [calculate] density 
        # TODO: implement later
        #n_tokens = len(tokenizer.tokenize(rac_text)) 
        #density = sum(ratings[answerable] >= threshold) / n_tokens
        #norm_density = (density / density_oracle) ** gamma

        outputs['coverage'].append(coverage)
        outputs['coverage_at_10'].append(coverage_at_10)
        #outputs['density'].append(norm_density)
        outputs['num_segs'].append(len(docids))
        #outputs['num_tokens'].append(n_tokens)

    # results
    mean_coverage = np.mean(outputs['coverage'])
    mean_coverage_at_10 = np.mean(outputs['coverage_at_10'])
    #mean_density = np.mean(outputs['density'])
    mean_num_segments = np.mean(outputs['num_segs'])
    #mean_num_tokens = np.mean(outputs['num_tokens'])
    num_coverage = len(outputs['coverage'])

    output_eval = {
        'mean_coverage': float(mean_coverage),
        'mean_coverage_at_10': float(mean_coverage_at_10),
        #'mean_density': mean_density,
        'mean_num_segments': float(mean_num_segments),
        #'mean_num_tokens': mean_num_tokens,
        'num_coverage': num_coverage,
    }

    # results from ir_measures if have runs
    runs = get_runs_from_rac_data(rac_data)
    if runs is not None:

        rank_results = ir_measures.calc_aggregate([alpha_nDCG@20], diversity_qrels, runs)
        output_eval['alpha_nDCG'] = rank_results[alpha_nDCG@20]

        """ TODO: implement other metrics later
        qrels = binarize(qrels)

        rank_results = ir_measures.calc_aggregate([R@100, MAP@100, nDCG@100], qrels, runs)
        output_eval['Recall@100'] = rank_results[R@100] 
        output_eval['MAP@100'] = rank_results[MAP@100]
        output_eval['nDCG@100'] = rank_results[nDCG@100]

        runs = sort_and_truncate(runs, max_k) 
        rank_results = ir_measures.calc_aggregate([R@100, MAP, nDCG], qrels, runs)
        output_eval['Recall'] = rank_results[R@100] 
        output_eval['MAP'] = rank_results[MAP]
        output_eval['nDCG'] = rank_results[nDCG]
        """

    return output_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--index_dir", type=str, default='data/index')
    args = parser.parse_args()

    rac_evaluate(
        corpus, qrels, judgements, diversity_qrels,
        rac_data,
        threshold=args.threshold,
        rel_threshold=args.rel_subset,
        runs=runs,
        tokenizer_name='bert-base-uncased',
        gamma=0.5, tag='experiment'
    )
