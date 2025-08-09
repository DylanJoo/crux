import gzip
import json
from glob import glob
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from ...tools import batch_iterator, normalize_text

def load_topic(split='train'):
    ds = load_dataset("corbyrosset/researchy_questions")[split]
    queries = {}
    for example in ds:
        queries[example['id']] = example['question']
    return queries

def load_queries(split='train'):
    ds = load_dataset("corbyrosset/researchy_questions")[split]
    queries = {}
    for example in ds:
        queries[example['id']] = example['question']
        for i, query in enumerate(example['gpt4_decomposition']['queries']):
            queries[example['id'] + f'::{i}'] = query
    return queries

def load_subtopics(split='train', head_only=False):
    from .text_utils import remove_header
    ds = load_dataset("corbyrosset/researchy_questions")[split]
    subquestions = {}
    for example in ds:
        if head_only:
            subquestions[example['id']] = [h[0] for h in example['gpt4_decomposition']['headers']]
        else:
            subquestions[example['id']] = \
                    [remove_header(q) for q in example['gpt4_decomposition']['subquestions']]
    return subquestions

def create_subset_corpus(
    glob_path="/export/common/data/corpora/clueweb22-b/txt/en/en00/en00*/en*json.gz", 
    subset=set(), 
    shard=0, num_shards=1
):
    files = glob(glob_path)
    shard_size = len(files) // num_shards
    files = files[shard * shard_size: (shard + 1) * shard_size]

    def process_file(file_path, docid_set=set()):
        document_list = []
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                id_field = "URL-hash" if "id" not in data.keys() else "id" 
                content_field = "Clean-Text" if "contents" not in data.keys() else "contents"
                if data[id_field] in docid_set:
                    document_list.append({
                        'id': data[id_field],
                        'title': data.get('title', ''),
                        'text': data.get('Clean-Text', data.get('URL', '')),
                        'url': data.get('URL', ''), 
                        'clueweb_id': data.get('ClueWeb22-ID', '')
                    })
        return document_list

    corpus = {}
    with ThreadPoolExecutor(32) as executor:
        for batch_files in tqdm(
            batch_iterator(files, 100), desc="Processing files", total=len(files) // 100+1
        ): 
            found_document_list = sum(
                    executor.map(
                        lambda x: process_file(x[0], x[1]), 
                        [(file_path, subset) for file_path in batch_files]),
                    []
            )

            for doc in found_document_list:
                corpus[doc['id']] = doc
                subset.discard(doc['id'])

    print(f"Total documents found: {len(corpus)}. Size of the subset documents: {len(subset)}")
    return corpus

def get_qrel():
    ds = load_dataset("corbyrosset/researchy_questions")['train']
    qrel = {}
    for example in ds:
        rel_docs = [item['CluewebURLHash'] for item in example['DocStream']]
        qrel[example['id']] = {doc: 1 for doc in rel_docs}
    return qrel

def load_corpus(**kwargs):
    raise NotImplementedError("This function is in `crux.tools.load_corpus`")
def load_run_or_qrel(path, topk=10, threshold=3):
    raise NotImplementedError("This function is in `crux.tools.load_run_or_qrel`")

def replace_tags(sent, tag='q'):
    if tag == 'q':
        sent = re.sub(r"\<q\>|\<\/q\>", "\n", sent)
    if tag == 'p':
        sent = re.sub(r"\<p\>|\<\/p\>", "\n", sent)
    pattern = re.compile(r"\n+")
    sent = re.sub(pattern, '\n', sent)
    return sent

def remove_private_use_chars(s):
    return ''.join(c for c in s if not ('\ue000' <= c <= '\uf8ff'))

def add_to_doc_outputs(doc_outputs: list, doc_output: str, n: int) -> list:
    doc_output = normalize_text(doc_output)
    if doc_output == " ":
        doc_outputs.append(["No content."])
    else:
        doc_output = doc_output.strip().split('</p>')[:n]
        doc_output = [replace_tags(o, 'p').strip() for o in doc_output]
        doc_output = [o.strip() for o in doc_output if o.strip() != ""]
        doc_outputs.append(doc_output)
    return doc_outputs

# Move all the preprocessing/postprocessing function to a separate file
# [TODO] use the tools/{dataset_name}/*.py to load the data
def load_passages(path: str, dataset_name: str, n=3, max_queries=None) -> list:
    if dataset_name == "researchy_questions":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        passages = []
        for i, item in tqdm(enumerate(data), total=len(data)):
            if max_queries and i > max_queries:
                break
            example_id = item['id']

            doc_outputs = []
            for doc_output in item['DocStream']:
                doc_output = remove_private_use_chars(doc_output["Snippet"])
                doc_outputs = add_to_doc_outputs(doc_outputs, doc_output, n)

            passages.append({
                "id": example_id, 
                "texts": doc_outputs, 
                "docs_full_texts": [normalize_text(d["CluewebURLHash"]) for d in item["DocStream"]] # TODO: adding file hash for now, should be the entire document content
            })
    else:
        # load data from one of the original CRUX paper datasets
        data = json.load(open(path, 'r'))

        passages = []
        for i, item in enumerate(data['data']):
            example_id = item['id']

            doc_outputs = []
            for doc_output in item['docs']['output']:
                doc_outputs = add_to_doc_outputs(doc_outputs, doc_output, n)

            passages.append({
                "id": example_id, 
                "texts": doc_outputs, 
                "docs_full_texts": [normalize_text(d) for d in item["docs"]["full_text"]]
            })
    
    return passages

def sort_and_truncate(run, topk=None):
    truncated_run = {}
    for qid, docid_scores in run.items():
        sorted_docs = dict(sorted(docid_scores.items(), key=lambda x: x[1], reverse=True)[:topk])
        truncated_run[qid] = sorted_docs
    return truncated_run

