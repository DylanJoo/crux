import os
import glob
from collections import defaultdict
import json

import pandas as pd

root_dir = os.environ.get('CRUX_ROOT', '/scratch/project_465001640/personal/dylan/datasets/crux')

def load_data():
    topic = load_topic()
    subtopics = load_subtopics("nuggets")
    qrel = get_qrel()

    data_list = []
    for id in topic:
        data_list.append({
            'id': id,
            'topic': topic[id],
            'subtopics': subtopics.get(id, None),
            'report': "NA",
            'qrel': qrel.get(id, None)
        })

    df = pd.DataFrame(data_list).dropna(axis=0)
    df = df.set_index('id')
    return df

def load_topic():
    path = os.path.join(root_dir, 'crux-neuclir/topic', 'neuclir24-test-request.jsonl')
    topics = {}
    with open(path, "r") as f:
        for i, line in enumerate(f):
            item = json.loads(line.strip())
            title = item['title']
            backgrpund = item["background"]
            problem_statement = item["problem_statement"]
            topics[str(item["request_id"])] = title + " " + problem_statement
    return topics

# NOTE: NeuCLIR use QA-level nugget with empty nugget number 
def load_subtopics(subset='nuggets'): 
    """ subset='nuggets' or 'subquestions' """
    subquestions = {}
    if subset == 'nuggets':
        file = os.path.join(root_dir, f"crux-neuclir", f"subtopics/{subset}.human.jsonl")
        items = [json.loads(l) for l in open(file).readlines()]
        subquestions.update({i['id']: i['nuggets'] for i in items})
    elif subset == 'subquestions':
        file = os.path.join(root_dir, f"crux-neuclir", f"subtopics/subquestions/{subset}.human.jsonl")
        items = [json.loads(l) for l in open(file).readlines()]
        subquestions.update({i['id']: i['subquestions'] for i in items})
    return subquestions

def get_qrel(tau=3):
    from ..generic.ir_utils import load_run_or_qrel
    path = os.path.join(root_dir, "crux-neuclir", "qrels/neuclir24-test-request.qrel")
    qrel = load_run_or_qrel(path, topk=1000, threshold=1)
    return qrel

def sort_and_truncate(run, max_k_dict=None):
    truncated_run = {}
    for qid, docid_scores in run.items():
        topk = max_k_dict[qid]
        sorted_docs = dict(sorted(docid_scores.items(), key=lambda x: x[1], reverse=True)[:topk])
        truncated_run[qid] = sorted_docs
    return truncated_run

def binarize(qrels):
    binarized_qrels = {}
    for qid, docid_scores in qrels.items():
        docid_scores = {docid: 1 for docid, score in docid_scores.items()}
        binarized_qrels[qid] = docid_scores
    return binarized_qrels

### NOTE: These are the deprecated function? (to be confirmed)
# def load_subtopics_human(path,
#                          raw_topics=None,
#                          create_new_subtopics=False):
#     # [TODO] AND/OR in the pipeline
#     files = [f for f in glob.glob(f"{path}/nuggets_*json")]
#     subquestions = {}
#     for file in files:
#         match = re.search(r"nuggets_(\d+)\.json$", file)
#         qid = str(match.group(1))
#         data = json.load(open(file, "r"))
#         subquestions[qid] = list(data.keys())
#     return subquestions

# def get_judgements_path(args):
#     """judgements are computed by running augmentation/gen_ratings.py"""
#     judgements_dir = os.path.join(args.crux_dir, args.dataset_name, args.tag)
#     path = os.path.join(judgements_dir, f"crux_{args.model}.jsonl")
#     return path

# def load_queries(path, fields=['title', 'problem_statement']):
#     queries = {}
#     with open(path, "r") as f:
#         for line in f:
#             data = json.loads(line.strip())
#             queries[data.pop('request_id')] = " ".join([data[field] for field in fields])
#     return queries

# def load_diversity_qrels(path: str) -> list:
#     qrels = pd.read_csv(path, sep="\s+", names=["query_id", "iteration", "doc_id", "relevance"])
#     diversity_qrels = [Qrel(str(row.query_id), row.doc_id, row.relevance, row.iteration) for row in qrels.itertuples(index=False)]
#     return diversity_qrels

# def load_ratings(path):
#     ratings = defaultdict(lambda: defaultdict(lambda: None))
#     contexts = defaultdict(lambda: None)
#     if os.path.exists(path):
#         with open(path, 'r') as f:
#             for line in f:
#                 data = json.loads(line.strip())
#                 id = data['id']
#                 ratings[id].update({data['pid']: data['rating']})
#     return ratings

# def load_subtopics(args, topic):
#     # TODO: replace with code from the auto-nuggetization team
#     subtopics = generate_subtopics(args, topic)
#     return subtopics

# async def async_load_subtopics(args, topic):
#     # [TODO] replace with code from the auto-nuggetization team
#     subtopics = await async_generate_subtopics(args, topic)
#     return subtopics

# def prepreocess(texts):
#     pattern = re.compile(r"^(\d+)*\.")
#     texts = re.sub(r"\<q\>|\<\/q\>", "\n", texts)
#     texts = re.sub(pattern, "\n", texts)
#     pattern = re.compile(r"^(\d+)*\.")
#     texts = re.sub(pattern, "", texts)
#     return texts
# 
# def load_corpus_online(path=None):
#     class get_content:
#         def __len__(self):
#             return 1
# 
#         def __getitem__(self, doc_id):
#             doc = requests.post(url="http://10.162.95.158:5000", json={"collection": "neuclir", "id": doc_id}).json()
#             text = doc.get("text", "").replace("\n", " ").strip()
#             title = doc.get("title", "").replace("\n", " ").strip()
#             return {"text": text, "title": title}
# 
#     return get_content()

# def load_nuggets(path, include_answer=False):
#     if os.path.isdir(path):
#         files = [f for f in glob.glob(f"{path}/*")]
#     else:
#         files = [path]
# 
#     nuggets = {}
#     for file in files:
#         match = re.search(r"nuggets_(\d+)\.json$", file)
#         if match:
#             qid = str(match.group(1))
#             nuggets[qid] = json.load(open(file, "r"))
#     return nuggets

