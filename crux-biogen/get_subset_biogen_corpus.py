import os
import json
from crux.tools import load_run_or_qrel
from crux.tools.biogen.ir_utils import load_topic, load_subset_corpus

pool_file = '/home/hltcoe/jhueiju/temp/biogen/runs/pool1k.txt'
output_jsonl='/home/hltcoe/jhueiju/temp/biogen/subset_corpus.jsonl'
corpus_dir='/exp/scale25/biogen/docs_from_track/jsonl_collection'
topic = load_topic()

docids = []
with open(pool_file, 'r') as f:
    for line in f:
        docids.append( line.strip() )

corpus = load_subset_corpus(os.path.join(corpus_dir, "*.jsonl"), subset=set(docids))

with open(output_jsonl, 'w') as f:
    for docid, doc in corpus.items():
        json.dump(doc, f)
        f.write('\n')
