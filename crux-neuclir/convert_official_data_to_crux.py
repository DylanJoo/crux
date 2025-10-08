import re
import os
import json
from glob import glob
from collections import defaultdict

## 1. nugget to subtopic
root_dir = '/users/judylan1/datasets/crux'
nugget_dir = f'{root_dir}/crux-neuclir/nuggets'

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
            for answer in type_and_answers[1]: # so the list represents acceptable answers
                answers += (answer,)
                evidences = type_and_answers[1][answer]

                ## debug 
                if (len(evidences) == 0) or (answer.strip() == ""):
                    print(f"# [{qid}] [QA(OR)] {question}->{answer} [#evidence] {len(evidences)}")

            subquestions_with_answer[qid].append( (question, answers) )

        if type_and_answers[0] == 'AND': 
            for answer in type_and_answers[1]: # so the each answer are separated
                subquestions_with_answer[qid].append( (question, answer) )
                evidences = type_and_answers[1][answer]

                ## debug 
                if (len(evidences) == 0) or (answer.strip() == ""):
                    print(f"# [{qid}] [QA (AND)] {question}->{answer} [#evidence] {len(evidences)}")

# subquestions
with open(f'{root_dir}/crux-neuclir/subtopics/subquestions.human.jsonl', 'w') as f:
    for qid in subquestions:
        if qid not in ['324', '361', '387']:
            f.write(json.dumps({'id': qid, 'subquestions': subquestions[qid]})+'\n')

# nuggets
with open(f'{root_dir}/crux-neuclir/subtopics/nuggets.human.jsonl', 'w') as f:
    for qid in subquestions_with_answer:
        if qid not in ['324', '361', '387']:
            f.write(json.dumps({'id': qid, 'nuggets': subquestions_with_answer[qid]})+'\n')


## 2. qrel to ratings
# ratings = {}
# with open(f"{root_dir}/crux-neuclir/qrels/neuclir24-all-request.qrel") as f:
#     for line in f:
#         qid, iteration, docid, rel = line.strip().split()
#         ratings[qid].update({docid: float(rel)})

#
# with open(f"{root_dir}/crux-neuclir/judge/ratings.human.jsonl") as f:
#     for qid in qrel:
#         for docid in qrel[qid]:
