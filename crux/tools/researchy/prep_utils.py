import os
import gzip
import json
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from ...tools import batch_iterator

def create_subset_corpus(
    glob_path="/export/common/data/corpora/clueweb22-b/txt/en/en00/en00*/en*json.gz", 
    subset=set(), 
    shard=0, num_shards=1
):
    files = sorted(glob(glob_path))
    print('Total files:', len(files))
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
