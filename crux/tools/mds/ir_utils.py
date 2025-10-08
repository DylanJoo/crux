import os
import json
from glob import glob
from datasets import load_dataset, load_from_disk
import pandas as pd

root_dir = os.environ.get('CRUX_ROOT', '/scratch/project_465001640/personal/dylan/datasets/crux')

def load_data(subset='multi_news'):
    topic = load_topic(subset)
    subtopics = load_subtopics(subset)
    report = load_report(subset)
    qrel = get_qrel(subset)

    data_list = []
    for id in topic:
        data_list.append({
            'id': id,
            'topic': topic[id],
            'subtopics': subtopics.get(id, None),
            'report': report.get(id, None),
            'qrel': qrel.get(id, None)
        })

    df = pd.DataFrame(data_list).dropna(axis=0)
    df = df.set_index('id')
    return df

def load_topic(subset='multi_news'):
    path = os.path.join(root_dir, f"crux-mds-{subset}", f"topic/requests.*.jsonl")
    topic = {}
    for file in glob(path):
        items = [json.loads(l) for l in open(file).readlines()]
        topic.update({i['id']: i['request'] for i in items})
    return topic

# TODO: consider update the hf dataset with subotopics
def load_subtopics(subset='multi_news'):
    path = os.path.join(root_dir, f"crux-mds-{subset}", "subtopics/subquestions.*.jsonl")
    subquestions = {}
    for file in glob(path):
        items = [json.loads(l) for l in open(file).readlines()]
        subquestions.update({i['id']: i['subquestions'] for i in items})
    return subquestions

def load_report(subset='multi_news'):
    if subset == 'multi_news':
        ds = load_multi_news()['test']
    if subset == 'duc04':
        ds = load_duc04()
    # 
    reports = {}
    for example in ds:
        reports[f"{example['id']}"] = example['summary']
    return reports

# NOTE: the split is not used.
def get_qrel(subset='multi_news', tau=3):
    from ..generic.ir_utils import load_run_or_qrel
    path = os.path.join(root_dir, f"crux-mds-{subset}", f"qrels/div_qrels-tau{tau}.txt")
    qrel = load_run_or_qrel(path, topk=1000, threshold=1)
    return qrel

def get_rating(subset='multi_news', split='test'):
    from ..generic.ir_utils import load_ratings
    dir = os.path.join(root_dir, f"crux-mds-{subset}/judge")
    ratings = load_ratings(dir)
    return ratings

##### Load from the origianl data, and with the preprocessing
def load_multi_news(load_from_source=False):
    if load_from_source:
        from .text_utils import parse_mds, normalize_list, flatten_and_normalize, maybe_chunking
        from huggingface_hub import snapshot_download
        repo_path = snapshot_download(repo_id="DylanJHJ/crux-mds", repo_type='dataset')
        ds = load_from_disk(repo_path+'/sources/multi_news')
        ds = ds.map(lambda x: {"long_document": parse_mds(x['document'])})
        ds = ds.filter(lambda x: len(x['long_document']) >=2 )
        ds = ds.map(lambda x: {"document": maybe_chunking(x['long_document'], n=1024)})
        for split in ds:
            ds[split] = ds[split].add_column("id", [f"multi_news-{split}-{i}" for i in range(len(ds[split]))])
        ds = ds.select_columns(['id', 'summary', 'document', 'long_document'])
    else:
        ds = load_dataset('DylanJHJ/crux-mds-multi_news-source')
    return ds

def load_duc04(load_from_source=False):
    if load_from_source:
        from .text_utils import parse_mds, normalize_list, flatten_and_normalize, maybe_chunking
        from huggingface_hub import snapshot_download
        repo_path = snapshot_download(repo_id="DylanJHJ/crux-mds", repo_type='dataset')
        ds = load_from_disk(repo_path+'/sources/duc04')['train']
        ds = ds.rename_column('context', 'long_document')
        ds = ds.map(lambda x: {
            "long_document": normalize_list(x['long_document']), 
            "summary": flatten_and_normalize(x['summary'])
        })
        ds = ds.filter(lambda x: len(x['long_document']) >=2 )
        ds = ds.map(lambda x: {"document": maybe_chunking(x['long_document'], n=1024)})
        temp_ids = ds['task_id']
        ds = ds.remove_columns("task_id")
        ds = ds.add_column("id", [f"duc04-test-{i}" for i in range(len(ds))])
        ds = ds.add_column("task_id", temp_ids)
        ds = ds.select_columns(['id', 'summary', 'document', 'long_document', 'task_id'])
    else:
        ds = load_dataset("DylanJHJ/crux-mds-duc04-source")['train']
    return ds

