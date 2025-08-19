import os
import json
from glob import glob
from datasets import load_dataset, load_from_disk
from .text_utils import (
    parse_mds,
    normalize_list,
    flatten_and_normalize,
    maybe_chunking
)

# TODO: consider update the hf dataset with subotopics
def load_topic(
    subset='multi_news', split='test', 
    root_dir='/users/judylan1/temp/datasets/crux'
):
    path = os.path.join(root_dir, f"crux-mds-{subset}", "topic")
    topic = {}
    for file in glob(path):
        items = [json.loads(l) for l in open(file).readlines()]
        topic.update({i['id']: i['request'] for i in items})
    return topic

# TODO: consider update the hf dataset with subotopics
def load_subtopics(
    subset='multi_news', split='test', 
    root_dir='/users/judylan1/temp/datasets/crux'
):
    path = os.path.join(root_dir, f"crux-mds-{subset}", "subtopics/*")
    subquestions = {}
    for file in glob(path):
        items = [json.loads(l) for l in open(file).readlines()]
        subquestions.update({i['id']: i['subquestions'] for i in items})
    return subquestions

def load_multi_news(load_from_source=False):
    if load_from_source:
        from huggingface_hub import snapshot_download
        repo_path = snapshot_download(repo_id="DylanJHJ/crux", repo_type='dataset')
        ds = load_from_disk(repo_path+'/sources/multi_news')
        ds = ds.map(lambda x: {"long_document": parse_mds(x['document'])})
        ds = ds.filter(lambda x: len(x['long_document']) >=2 )
        ds = ds.map(lambda x: {"document": maybe_chunking(x['long_document'], n=1024)})
        for split in ds:
            ds[split] = ds[split].add_column("id", [f"multi_news-{split}-{i}" for i in range(len(ds[split]))])
        ds = ds.select_columns(['id', 'summary', 'document', 'long_document'])
    else:
        ds = load_dataset('DylanJHJ/crux-mds-multi_news')
    return ds

def load_duc04(load_from_source=False):
    if load_from_source:
        from huggingface_hub import snapshot_download
        repo_path = snapshot_download(repo_id="DylanJHJ/crux", repo_type='dataset')
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
        ds = load_dataset("DylanJHJ/crux-mds-duc04")['train']
    return ds

def load_reports(subset='multi_news', split='test'):
    if subset == 'multi_news':
        ds = load_multi_news()[split]
    if subset == 'duc04':
        ds = load_duc04()
    # 
    reports = {}
    for example in ds:
        reports[example['id']] = example['summary']
    return reports

