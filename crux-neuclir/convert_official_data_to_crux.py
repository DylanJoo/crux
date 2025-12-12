import re
import os
import json
from glob import glob
from collections import defaultdict

root_dir = '/home/dju/datasets/crux'
nugget_dir = f'{root_dir}/crux-neuclir/nuggets'

## 1. nugget to subtopic
subquestions = defaultdict(list)
subquestions_with_answer = defaultdict(list)
for file in glob(os.path.join(nugget_dir, f'nuggets_???.json')):
    item = json.load(open(file))
    match = re.search(r"nuggets_(\d+)\.json$", file)
    qid = str(match.group(1))

    for question, type_and_answers in item.items():
        subquestions[qid].append( question )

        if type_and_answers[0] == 'OR':
            answers = ()

            evidences_count = 0
            for answer in type_and_answers[1]: # so the list represents acceptable answers
                evidences = type_and_answers[1][answer]
                answers += (answer,)
                evidences_count += len(evidences)

                ## debug 
                if (len(evidences) == 0) or (answer.strip() == ""):
                    print(f"# [{qid}] [QA(OR)] {question}->{answer} [#evidence] {len(evidences)}")

            if evidences_count > 0: # For OR case, ignore the nugget question, if none of the answers has evidence, 
                subquestions_with_answer[qid].append( (question, answers) )

        if type_and_answers[0] == 'AND': 

            evidences_count = 0
            for answer in type_and_answers[1]: # so the each answer are separated
                evidences = type_and_answers[1][answer]
                subquestions_with_answer[qid].append( (question, answer) )
                evidences_count += len(evidences)

                ## debug 
                if (len(evidences) == 0) or (answer.strip() == ""):
                    print(f"# [{qid}] [QA(AND)] {question}->{answer} [#evidence] {len(evidences)}")

            if evidences_count < 0: # For AND case, ignore the nugget quesion only when all the evidences of answers are unavilable.
                print(f"Warning. All answers of this nugget question have no evidences.")

## subquestions
with open(f'{root_dir}/crux-neuclir/subtopics/subquestions.human.jsonl', 'w') as f:
    for qid in subquestions:
        if qid not in ['324', '361', '387']:
            f.write(json.dumps({'id': qid, 'subquestions': subquestions[qid]})+'\n')

with open(f'{root_dir}/crux-neuclir/subtopics/nuggets.human.jsonl', 'w') as f:
    for qid in subquestions_with_answer:
        if qid not in ['324', '361', '387']:
            f.write(json.dumps({'id': qid, 'nuggets': subquestions_with_answer[qid]})+'\n')


## 2. qrel to ratings
"""
This script transforms the official qrel into the flatten query-document ratgins scores. 
The empry nuggets are done by purpopse.
"""
from crux.tools import load_diversity_qrel
qrel = load_diversity_qrel(f"{root_dir}/crux-neuclir/qrels/neuclir24-all-request.qrel")
max_iteration_mapping = json.loads(qrel.groupby(['query_id']).iteration.agg(max).to_json())

iteration_mapping = qrel.groupby(['query_id', 'doc_id']).iteration.apply(list).to_dict()
with open(f"{root_dir}/crux-neuclir/judge/ratings.human.jsonl", 'w') as f:
    for (qid, docid), labels in iteration_mapping.items():
        max_iter = max_iteration_mapping[str(qid)]
        rating = [0 for _ in range(max_iter)]

        for label in labels:
            rating[label-1] = 3

        f.write(json.dumps({'id': str(qid), 'docid': docid, 'rating': rating})+'\n')
