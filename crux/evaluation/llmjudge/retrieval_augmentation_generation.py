""" Fix the long context """
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import re
import argparse
import json
import numpy as np
from tqdm import tqdm
from glob import glob
# from transformers import AutoTokenizer
# from tools import remove_citations

# Define the prompt for rating generation
prompt_template = """\
Instruction: Determine whether the question can be answered based on the provided context. Rate the context on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating.

Guideline: 
- 5: The context is highly relevant, complete, and accurate to the question.
- 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies.
- 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies.
- 2: The context has limited relevance and completeness, with significant gaps or inaccuracies.
- 1: The context is minimally relevant or complete, with substantial shortcomings.
- 0: The context is not relevant or complete at all.

Question: {question}
Context: {context} 

Rating: """

def llm_judgement(llm, response, questions):

    prompt = [
            prompt_template.format(
            question=question,
            context=response,
        ) for question in questions
    ]

    output = llm.generate(prompt, max_tokens=5, min_tokens=1)
    output = [re.sub(r'\[\d+\]', '', o).strip() for o in output] # remove citation
    output = [o.replace("<|im_end|>", "").rstrip() for o in output]

    # extract rating
    pattern = re.compile(r"\d|-\d")
    output = [re.findall(pattern, o + "-1")[0] for o in output]
    output = [-1 if len(o) == 0 else int(o) for o in output]
    return output

def rag_evaluate(
    generator, 
    corpus, qrels, judgements, 
    rag_data,
    questions,
    threshold=0,     # answerability threshold (tau)
    rel_threshold=3, # on qrel's last column
    tokenizer_name='meta-llama/Llama-3.1-8B-Instruct',
    gamma=0.5,
    used_field='response'
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    outputs = {'coverage': [], 'density': [], 'num_segs': [], 'num_tokens': []}

    # filtered qrels by (1) rag_data and also (2) 
    overlapped = {k: v for k, v in qrels.items() if k in rag_data}

    if len(overlapped) != len(qrels):
        logger.warning(' #Topics in qrels and rag_data are not consistent.' + \
                f' Got {len(qrels)} and {len(rag_data)}.')
        qrels = overlapped

    for qid in tqdm(qrels, desc='RAG Evaluating', total=len(qrels)):

        # [oracle] 
        docids = [docid for docid, score in qrels[qid].items() if score >= rel_threshold ] 
        rac_text = " ".join([corpus[docid]['text'] for docid in docids])
        n_tokens = len(tokenizer.tokenize(rac_text))

        judgement_oracle = np.array([judgements[qid][docid] for docid in docids]).max(0)
        answerable = (judgement_oracle >= threshold)
        density_oracle = sum(answerable) / n_tokens

        # [retrieval-augmented generation]
        # [TODO] add the extract_citation function
        rag_text = remove_citations(rag_data[qid][used_field]) 
        ratings = np.array(llm_judgement(generator, rag_text, questions[qid]))
        print(f'{qid}#rating:', ratings)

        # [calculate] coverage
        coverage = sum(ratings[answerable] >= threshold) / sum(answerable)

        # [calculate] density
        n_tokens = len(tokenizer.tokenize(rag_text)) 
        density = sum(ratings[answerable] >= threshold) / n_tokens
        norm_density = (density / density_oracle) ** gamma

        outputs['coverage'].append(coverage)
        outputs['density'].append(norm_density)
        outputs['num_tokens'].append(n_tokens)

    # results
    mean_coverage = np.mean(outputs['coverage'])
    mean_density = np.mean(outputs['density'])
    mean_num_tokens = np.mean(outputs['num_tokens'])
    num_coverage = len(outputs['coverage'])

    # print(outputs['coverage'])
    output_eval = {
        'mean_coverage': mean_coverage,
        'mean_density': mean_density,
        'mean_num_tokens': mean_num_tokens,
        'num_coverage': num_coverage,
    }
    return output_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    # data
    parser.add_argument("--topic_file", type=str, default='data/topics')
    parser.add_argument("--corpus_dir", type=str, default='data/corpus')
    parser.add_argument("--qrels_file", type=str, default='data/qrels')
    parser.add_argument("--judgement_file", type=str, default='data/judgements')
    # evaluate & model 
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--num_gpus", type=int, default=1)
    # rag output
    parser.add_argument("--result_jsonl", type=str, default='testing.jsonl')
    # controls
    parser.add_argument("--used_field", type=str, default='response')
    args = parser.parse_args()


    # model
    from generate.llm.utils import check_if_ampere
    if check_if_ampere:
        if args.num_gpus > 1:
            from generate.llm.vllm_api import LLM
        else:
            from generate.llm.vllm_back import LLM
    else:
        from generate.llm.hf_back import LLM
    generator = LLM(
        model=args.model_name_or_path,
        top_p=1,
        temperature=0 if check_if_ampere else 1e-10,
        num_gpus=args.num_gpus,
        gpu_memory_utilization=0.9
    )

    # data
    from tools import load_topics, load_questions, load_corpus, load_qrels, load_judgements
    topics = load_topics(args.topic_file)
    questions = load_questions(args.topic_file)
    corpus = load_corpus(args.corpus_dir)
    qrels = load_qrels(args.qrels_file)
    judgements = load_judgements(args.judgement_file)

    output_rac = {}
    with open(args.result_jsonl, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            qid = data['qid']
            output_rac[qid] = data

    # evaluation
    from evaluation import rag_evaluate
    output_rag_eval = rag_evaluate(
        generator=generator,
        corpus=corpus, 
        qrels=qrels, 
        judgements=judgements,
        rag_data=output_rac,
        questions=questions,
        threshold=args.threshold,
        tokenizer_name=args.model_name_or_path,
        gamma=args.gamma,
        used_field=args.used_field
    )

    metrics = ['final_coverage', 'final_density']
    values =  [str(output_rag_eval['mean_coverage']), str(output_rag_eval['mean_density'])]
    print(" ".join(['RAG-eval'] + metrics))
    print(" ".join(['##' + args.result_jsonl] + values))
