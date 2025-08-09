""" The reranking pipeline is copied from Qwen3-Reranker-0.6B on Huggingface """
from typing import Dict, Optional, List

import json
import logging

import torch

from transformers import AutoTokenizer, is_torch_npu_available
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import math
from vllm.inputs.data import TokensPrompt

import os
import argparse
from tqdm import tqdm

def format_instruction(instruction, query, doc):
    text = [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]
    return text

def process_inputs(tokenizer, pairs, instruction, max_length, suffix_tokens):
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    messages =  tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    messages = [ele[:max_length] + suffix_tokens for ele in messages]
    messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
    return messages

def compute_logits(model, messages, sampling_params, true_token, false_token):
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]
        token_count = len(outputs[i].outputs[0].token_ids)
        if true_token not in final_logits:
            true_logit = -10
        else:
            true_logit = final_logits[true_token].logprob
        if false_token not in final_logits:
            false_logit = -10
        else:
            false_logit = final_logits[false_token].logprob
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        score = true_score / (true_score + false_score)
        scores.append(score)
    return scores

def rerank(batch_size, shard, total_shards):

    # prepare model and tokenizer
    number_of_gpu = torch.cuda.device_count()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-0.6B')
    model = LLM(model='Qwen/Qwen3-Reranker-0.6B', tensor_parallel_size=number_of_gpu, max_model_len=10000, enable_prefix_caching=True, gpu_memory_utilization=0.9)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    max_length=8192
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    sampling_params = SamplingParams(temperature=0, 
        max_tokens=1,
        logprobs=20, 
        allowed_token_ids=[true_token, false_token],
    )

    # prepare corpus and runs
    from crux.tools import load_corpus, load_run_or_qrel, batch_iterator
    from crux.tools.researchy.ir_utils import load_subtopics, load_queries
    run = load_run_or_qrel('/exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-init-q.bm25.clueweb22-b.txt', 
                           topk=100)
    corpus = load_corpus('/exp/scale25/artifacts/crux/crux-researchy/docs/cw22-b-researchy-v1/corpus.pkl')
    subquestions = load_subtopics()
    queries = load_queries()
    print(f"Data loading complete. Number of queries: {len(run)}")

    qids = list(run.keys())
    qids = sorted(qids)
    shard_size = len(qids) // total_shards + 1
    qids = qids[shard * shard_size: (shard + 1) * shard_size]

    # ignore the qids that have been done
    output_run = f'/exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-init-q.bm25+qwen3.clueweb22-b.txt{shard}'
    if os.path.exists(output_run):
        run_done = load_run_or_qrel(output_run, topk=100)
        qids = [qid for qid in qids if qid not in run_done]
    print(f"Processing {len(qids)} queries in shard {shard}/{total_shards}.")

    ## prepare multiquery
    task = 'Given the list of questions as query, retrieve relevant passages that answer the questions.'
    new_queries = {}
    for qid in qids:
        new_query = ", ".join([queries[qid]] + subquestions[qid][:10])
        new_query = "[" + new_query + "]"
        new_queries[qid] = new_query

    ## Get documents
    documents = {}
    for qid in qids:
        documents[qid] = [corpus[docid]['text'] for docid in run[qid]]

    ## Pairs of ids and query-document
    all_scores = []
    for qid in tqdm(qids, desc=f"Processing queries in shard {shard}/{total_shards}"):
        query = new_queries[qid]
        pairs = [(query, " ".join(document.split()[:5000])) for document in documents[qid]]
        inputs = process_inputs(tokenizer, pairs, task, max_length-len(suffix_tokens), suffix_tokens)
        scores = compute_logits(model, inputs, sampling_params, true_token, false_token)
        # print(f'scores for {qid}: {scores}')

        # sort and covert score into run file
        with open(output_run, 'a') as f:
            docid_score_pairs = sorted(zip(run[qid], scores), key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(docid_score_pairs, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} BM25+qwen3-0.6b\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--shard', type=int, default=0, help='Shard index for distributed processing')
    parser.add_argument('--total_shards', type=int, default=20, help='Total number of shards for distributed processing')
    args = parser.parse_args()

    rerank(batch_size=args.batch_size, shard=args.shard, total_shards=args.total_shards)
    destroy_model_parallel()
