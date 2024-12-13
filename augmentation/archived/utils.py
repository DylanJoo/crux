import os
import re
import argparse
import json
from tqdm import tqdm
from glob import glob
from collections import defaultdict

def remove_citations(sent):
    sent = sent.strip()
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def replace_citations(sent):
    sent = re.sub(r"\[\d+\]", "\n", sent)
    pattern = re.compile(r"\n+")
    sent = re.sub(pattern, '\n', sent)
    return sent

def replace_tags(sent, tag='q'):
    if tag == 'q':
        sent = re.sub(r"\<q\>|\<\/q\>", "\n", sent)
    if tag == 'p':
        sent = re.sub(r"\<p\>|\<\/p\>", "\n", sent)

    pattern = re.compile(r"\n+")
    sent = re.sub(pattern, '\n', sent)
    return sent

def normalize_texts(texts, size=10000):
    texts = remove_citations(texts)
    texts = texts.strip()
    pattern = re.compile(r"\n")
    texts = re.sub(pattern, ' ', texts).strip()
    pattern = re.compile(r"\s+")
    texts = re.sub(pattern, ' ', texts).strip()
    texts = maybe_truncation(texts, size)
    return texts

def maybe_truncation(text, size=10000):
    if len(text) > size:
        words = text.split(' ')[:(size//10)]
        return " ".join(words)
    else:
        return text

def load_passages(path, n=3):
    data = json.load(open(path, 'r'))

    passages = []
    for i, item in enumerate(data['data']):
        example_id = item['example_id']

        doc_outputs = []
        for doc_output in item['docs']['output']:
            doc_output = normalize_texts(doc_output)
            # doc_output = replace_tags(doc_output, 'p')
            if doc_output == " ":
                doc_outputs.append(["No content."])

            else:
                doc_output = doc_output.strip().split('</p>')[:n]
                doc_output = [replace_tags(o, 'p').strip() for o in doc_output]
                doc_output = [o.strip() for o in doc_output if o.strip() != ""]
                doc_outputs.append(doc_output)

        passages.append({
            "example_id": example_id, 
            "texts": doc_outputs, 
            "docs_full_texts": [normalize_texts(d) for d in item["docs"]["full_text"]]
        })
    return passages

def load_question(path, n=10):
    data = json.load(open(path, 'r'))

    questions = []
    for i, item in enumerate(data['data']):
        example_id = item['example_id']
        if not isinstance(item['output'], list):
            outputs = item['output'].strip().split('</q>')[:n]
            outputs = [replace_tags(o).strip() for o in outputs]
            questions.append({"example_id": example_id, "texts": outputs})
    return questions

def load_topic(path):
    data = json.load(open(path, 'r'))

    topic = []
    for i, item in enumerate(data['data']):
        example_id = item['example_id']
        if not isinstance(item['output'], list):
            output = item['output'].strip().split('</r>')[0]
            output = normalize_texts(output)
            topic.append({"example_id": example_id, "texts": output})
    return topic


# def load_statements(path, n=10):
#     data = json.load(open(path, 'r'))
#
#     passages = []
#     for i, item in enumerate(data['data']):
#         example_id = item['example_id']
#
#         doc_outputs = []
#         for doc_output in item['doc_output']:
#             doc_output = replace_citations(doc_output)
#             if doc_output == "":
#                 doc_outputs.append(["No content."])
#
#             else:
#                 doc_output = doc_output.split('\n')
#                 doc_output = [o.strip() for o in doc_output if o.strip() != ""]
#                 doc_outputs.append(doc_output)
#
#         passages.append({"example_id": example_id, "texts": doc_outputs})
#     return passages

def load_nuggets_and_claims(path, n=10):
    data = json.load(open(path, 'r'))

    nuggets = []
    claims = []
    for i, item in enumerate(data['data']):
        example_id = item["example_id"]
        if len(item['output']) == 0:
            continue
        outputs = replace_citations(item['output'][0]).split('\n')[:n]
        ndocs = item['number_of_documents']
        nuggets.append({
            "example_id": example_id, 
            "contents": [normalize_texts(n).strip() for n in outputs],
            "full_text": normalize_texts(item['full_text'][0]), # we dont need this
            "type": 'nuggets',
            "ndocs": ndocs
        })

        for j in range(1, ndocs+1): # the first one is nuggets
            example_doc_id = f"{example_id}#{j}" 
            outputs = item['output'][j].strip().split('\n')[:n]
            claims.append({
                "example_doc_id": example_doc_id, 
                "contents": [normalize_texts(n).strip() for n in outputs],
                "full_text": normalize_texts(item['full_text'][j]),
                "type": 'claims',
            })
    return nuggets, claims

def load_result_to_dict(path, n=10, threshold=-99):
    results = defaultdict(list)

    with open(path, 'r') as f:
        for line in f:
            query_id, example_doc_claim_id, rank, score = line.split()
            example_id = query_id.rsplit(':', 1)[0]
            rank = int(rank)
            score = float(score)
            # type_of_premise = "nugget" if ":" in query_id else "fulltext"

            if len(results[query_id]) >= n:
                continue

            # for the concentrator. highere score and in the same example
            if (score >= threshold) and (example_id in example_doc_claim_id): 
                results[query_id].append({
                    "id": example_doc_claim_id, "rank": rank, "score": score
                })

            # for the concentrator. lower score but not in the example
            if (score < threshold) and (example_id not in example_doc_claim_id): 
                results[query_id].append({
                    "id": example_doc_claim_id, "rank": rank, "score": score
                })

    return results
