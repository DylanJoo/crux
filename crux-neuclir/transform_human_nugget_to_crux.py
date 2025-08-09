import json
from neuclir.ir_utils import (
    load_subtopics_human,
    load_runs_or_qrels
)


questions_all = load_subtopics_human('/exp/scale25/neuclir/eval/nuggets')
qrels = load_run_or_qrels('/exp/scale25/artifacts/crux/crux-neuclir/neuclir24-all-request.jsonl', threshold=3)

print(questions_all.keys())
print(qrels.keys())

with open('/exp/scale25/artifacts/crux/crux-neuclir/neuclir24-all-request.jsonl') as f:
    pass
