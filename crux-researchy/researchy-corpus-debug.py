from crux.tools import load_corpus
from crux.tools.researchy.ir_utils import get_qrel

corpus = load_corpus("/exp/scale25/artifacts/crux/crux-researchy/docs/qrel.doc00.jsonl") 
qrel = get_qrel() 

missing = set()
for qid in qrel:
    for docid in qrel[qid]:
        if docid not in corpus:
            missing.add(docid)

with open('missing_in_qrel.txt', 'w') as f:
    for docid in missing:
        f.write(f"Missing docid of qrel: {docid}\n")
