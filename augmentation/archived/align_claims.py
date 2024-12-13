import torch
import os
import argparse
import logging
from tqdm import tqdm
from glob import glob
from operator import itemgetter
from collections import defaultdict
from utils import load_nuggets_and_claims, load_question, maybe_truncation

from pyserini.search.lucene import LuceneSearcher
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None
AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"
OFFLOAD_HF="/ivi/ilps/personal/dju/temp/offload/"

def mine_distractor(
    query, 
    searcher, 
    k=10,
    max_docs_return=3,
    ignored_prefix=""
):

    hits = searcher.search(query, k)
    hit_doc_ids = []

    # filter the ignored
    ### schema: {example_id}#{n_doc}:{n_claims}
    for h in hits:
        if h.docid.split("#")[0] != ignored_prefix:
            hit_doc_ids.append(h.docid.split(":")[0])
            if len(hit_doc_ids) >= max_docs_return:
                return hit_doc_ids

    return []

@torch.no_grad()
def _run_nli_autoais(nugget, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(nugget, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)

    outputs = autoais_model.generate(
        input_ids, 
        max_new_tokens=10,
        return_dict_in_generate=True, 
        output_scores=True
    )

    result = autoais_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0

    o1 = outputs.scores[0][0].log_softmax(-1)
    o2 = outputs.scores[1][0].log_softmax(-1)
    entail_score = (o1[209] + o2[1])
    contra_score = (o1[3] + o2[632])
    ## [NOTE] see if we need to do the re-softmax trick.
    # score = torch.tensor([contra, entail]).softmax(-1).detach().numpy().tolist()[1]
    del outputs
    torch.cuda.empty_cache()

    return inference, entail_score

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-1}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

def compute_claims(nugget, claims):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS_MODEL, 
	    torch_dtype=torch.bfloat16, 
            device_map="cuda",
        )
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info("Computing claims...")
    scores = []
    predictions = []
    for i, claim in enumerate(claims):
        entail, score = _run_nli_autoais(nugget, claim)
        predictions.append(entail)
        scores.append(score)
        if i <= 10:
            logger.info(f"Example: premise: {nugget} hypothesis: {claim} | TRUE score: {score}")

    return predictions, scores

def write_results(writer, qid, docids, scores):
    result = {docid: score for docid, score in zip(docids, scores)}
    sorted_result = {k: v for k,v in sorted(result.items(), key=itemgetter(1), reverse=True)} 

    for i, (docid, score) in enumerate(sorted_result.items()):
        writer.write(f"{qid} {docid} {str(i+1)} {score}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--result_jsonl", type=str, default=None)
    parser.add_argument("--doc_claims_index", type=str, default=None)
    parser.add_argument("--premise_from", type=str, default='full_text', choices=['full_text', 'separated_nugget'])
    args = parser.parse_args()

    os.makedirs(args.result_jsonl.rsplit('/', 1)[0], exist_ok=True)
    writer = open(args.result_jsonl, 'w')

    # questions
    questions_all = []
    files_questions = glob(os.path.join(args.input_dir, "*question*.json"))
    for file in tqdm(files_questions):
        question = load_question(file) # one per example
        questions_all += question
    questions = {q['id']: q['contents'] for q in questions_all}

    # claims and nuggets
    claims_all = []
    nuggets_all = []
    files_claims = glob(os.path.join(args.input_dir, "*claims*.json"))
    for file in tqdm(files_claims):
        nuggets, claims = load_nuggets_and_claims(file)
        nuggets_all += nuggets
        claims_all += claims
    claims = {c['example_doc_id']: c['contents'] for c in claims_all}

    ## align claims to nugget
    start, end = 0, 0
    for nuggets_dict in tqdm(nuggets_all):
        example_id = nuggets_dict['example_id']
        nuggets = nuggets_dict['contents']
        summary = nuggets_dict['full_text']
        ndocs = nuggets_dict['ndocs']

        claim_ids = []
        claim_texts = []

        ### the potentially entailed claims from ndocs
        for n in range(1, 1+ndocs):
            example_doc_id = f"{example_id}#{n}"
            claim_ids += [f"{example_doc_id}:{i}" for i in range(len(claims[example_doc_id]))]
            claim_texts += claims[example_doc_id]

        ### the potentially contradicted claims (optional)
        if args.doc_claims_index is not None:
            for nugget in nuggets:
                searcher = LuceneSearcher(args.doc_claims_index)
                dis_example_doc_ids = mine_distractor(
                    query=nugget, 
                    searcher=searcher, 
                    k=10 if args.premise_from == 'full_text' else 1,
                    max_docs_return=1,
                    ignored_prefix=example_id
                )
                for example_doc_id in dis_example_doc_ids:
                    claim_ids += [f"{example_doc_id}:{i}" for i in range(len(claims[example_doc_id]))]
                    claim_texts += claims[example_doc_id]

        claim_scores, claim_predictions = [], []
        if args.premise_from == "full_text":
            query_id = example_id
            premise = maybe_truncation(summary, size=5000) # sometime would overlength
            predictions, scores = compute_claims(premise, claim_texts)
            claim_predictions += predictions
            claim_scores += scores
            write_results(writer, query_id, claim_ids, claim_scores)

        if args.premise_from == "separated_nugget":
            for j, premise in enumerate(nuggets):
                query_id = f"{example_id}:{j}" 
                predictions, scores = compute_claims(premise, claim_texts)
                claim_predictions += predictions
                claim_scores += scores
                write_results(writer, query_id, claim_ids, claim_scores)

                # clean the placeholder
                claim_scores = []
                claim_predictions = []

        del claim_scores, claim_predictions
        torch.cuda.empty_cache()

    writer.close()

