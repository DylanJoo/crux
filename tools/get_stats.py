import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re
import os
import yaml
import argparse
import json
import numpy as np
from glob import glob

from transformers import AutoTokenizer
from evaluation.llm_judge.utils import (
    load_topics, 
    load_questions, 
    load_contexts, 
    load_judgements,
    load_qrels
)

def get_token_length(entity, tokenizer):
    N = []
    for e in entity:
        N.append(len(tokenizer.tokenize(e)))
    return N

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to the config file")
    parser.add_argument("--split", type=str, default=None, help="Path to the config file")
    args = parser.parse_args()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-70B-Instruct')

    # entities
    topics = load_topics(
        os.path.join(args.dataset_dir, f"ranking_3/{args.split}_topics_exam_questions.jsonl")
    )
    questions = load_questions(
        os.path.join(args.dataset_dir, f"ranking_3/{args.split}_topics_exam_questions.jsonl"),
        n=10 if args.split == 'test' else 15
    )
    documents = load_contexts(os.path.join(args.dataset_dir, f"documents/{args.split}_docs.jsonl"))
    passages = load_contexts(os.path.join(args.dataset_dir, f"passages/{args.split}_psgs.jsonl"))
    # judgements = load_judgements(os.path.join(args.dataset_dir, f"ranking_3/{args.split}_judgements.jsonl"))
    qresl_3 = load_qrels(
        os.path.join(args.dataset_dir, f"ranking_3/{args.split}_topics_exam_questions.jsonl"),
        threshold=3
    )

    # tokenization and calculate
    topic_token_length = get_token_length(topics.values(), tokenizer)
    print("Topic\nAmount: {}\nAvg. length: {}\n".format(
        len(topic_token_length), round(np.mean(topic_token_length), 2)
    ))

    flatten_questions = [q for Q in questions.values() for q in Q]
    question_token_length = get_token_length(flatten_questions, tokenizer)
    print("Question\nAmount: {} ({})\nAvg. length: {}\n".format(
        len(questions), len(question_token_length), round(np.mean(question_token_length), 1)
    ))

    passage_token_length = get_token_length(passages.values(), tokenizer)
    print("Passage\nAmount: {}\nAvg. length: {}\n".format(
        len(passage_token_length), round(np.mean(passage_token_length), 1)
    ))

    document_token_length = get_token_length(documents.values(), tokenizer)
    print("Document\nAmount: {}\nAvg. length: {}\n".format(
        len(document_token_length), round(np.mean(document_token_length), 1)
    ))

    judgementV = get_token_length(documents.values(), tokenizer)
    print("Document\nAmount: {}\nAvg. length: {}\n".format(
        len(document_token_length), round(np.mean(document_token_length), 1)
    ))

if __name__ == "__main__":
    main()

