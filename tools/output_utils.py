from operator import itemgetter
from collections import defaultdict
from datasets import load_dataset
import json
import re
import unicodedata
from collections import Counter

def check_result(texts):
    texts = normalize_texts(texts)
    word_length = len(texts.split(' '))
    return {'word_length': word_length}

def normalize_texts(texts):
    texts = unicodedata.normalize('NFKC', texts)
    texts = texts.strip()
    pattern = re.compile(r"\n")
    texts = re.sub(pattern, ' ', texts).strip()
    pattern = re.compile(r"\s+")
    texts = re.sub(pattern, ' ', texts).strip()
    return texts

def load_hits_tsv(path):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            item = line.strip().split('\t')
            try:
                data[item[0] + item[1]].append({
                    "rank": int(item[2]),
                    "id": str(item[3]),
                    "qid": str(item[0]) + str(item[1]),
                    "target_contents": normalize_texts(item[4]),
                    "translation": normalize_texts(item[5])
                })
            except:
                print(item)
    return data

def load_hits_jsonl(path, key='id'):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            if ('unrelated.' in item[key]) or ('redundant.' in item[key]):
                continue
            try:
                item['rank'] = int(item['rank'])
                data[item['qid']].append(item)
            except:
                print(item)
    # rerank
    for qid in data:
        orig_item = data[qid]
        data[qid] = sorted(orig_item, key=itemgetter('rank'))
    return data


def citation_fixing(x):
    # cite with the word 'document'
    incorrect_citations = re.findall(r'Document \[\d+\]', x)
    for item in incorrect_citations:
        cite = re.findall(r'\[\d+\]', item)[0] # extract number
        x = x.replace(item, f"{cite}")

    # cite after punctuation --> cite before punc
    incorrect_citations = re.findall(r"\.\s*\[\d+\](?:\[\d+\])?", x)
    for item in incorrect_citations:
        cite = re.findall(r'\[\d+\](?:\[\d+\])?', item)[0] # extract number
        x = x.replace(item, f" {cite}.")
    return x


def citation_removal(x, return_numbers=False):
    # citations = re.findall(r'\[\d+\](?:\[\d+\])?', x)
    citations = re.findall(r'\[\d+\]', x)
    numbers = []
    for cite in citations:
        x = x.replace(cite, "").strip()
        numbers.append( cite.replace(r"[", "").replace(r"]", "") )

    numbers = [n for n, N in Counter(numbers).most_common()]
    if return_numbers:
        return x, numbers
    else:
        return x
