import json
import numpy as np
from compose_pairs import binary_rerank_greedy

data = [json.loads(d) for d in open('/home/dju/rag-rerank/data/mds/alignment/question_to_summaries_llama3.1.jsonl')]

o = binary_rerank_greedy(
    data[0],
    threshold=3
)
print(o)
