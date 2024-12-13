# Load evaluation data
from datasets import load_from_disk, concatenate_datasets
import re

def normalize_list(string_list):
    for i in range(len(string_list)):
        string_list[i] = normalize_text(string_list[i])
    return string_list

def flatten_and_normalize(string_list):
    string = " ".join(string_list)
    return normalize_text(string)

def normalize(string):
    string = string.strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile("</s>")
    string = re.sub(pattern, '|||||', string).strip() # align seperation 
    return string.split('|||||')

def normalize_text(string):
    string = string.strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    return string

def maybe_chunking(dlist, n=1024):
    overlength = [(i, len(d.split()) > n) for i, d in enumerate(dlist)]

    if any([o for _, o in overlength]):
        to_return = []
        for i, do_chunk in overlength:
            if do_chunk:
                words = dlist[i].split()
                while len(words) > 0:
                    to_return.append(" ".join(words[:512]))
                    words = words[512:]
            else:
                to_return.append(dlist[i])
        return to_return
    else:
        return dlist
def normalize_list(string_list):
    for i in range(len(string_list)):
        string_list[i] = normalize_text(string_list[i])
    return string_list

def flatten_and_normalize(string_list):
    string = " ".join(string_list)
    return normalize_text(string)

def normalize(string):
    string = string.strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile("</s>")
    string = re.sub(pattern, '|||||', string).strip() # align seperation 
    return string.split('|||||')

def normalize_text(string):
    string = string.strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    return string

def maybe_chunking(dlist, n=1024):
    overlength = [(i, len(d.split()) > n) for i, d in enumerate(dlist)]

    if any([o for _, o in overlength]):
        to_return = []
        for i, do_chunk in overlength:
            if do_chunk:
                words = dlist[i].split()
                while len(words) > 0:
                    to_return.append(" ".join(words[:512]))
                    words = words[512:]
            else:
                to_return.append(dlist[i])
        return to_return
    else:
        return dlist


from datasets import load_from_disk
# 
multi_news = load_from_disk("/home/dju/datasets/multi_news")['train']
multi_news = multi_news.map(lambda x: {
    "document": normalize(x['document']), 
    'mds-source': 'multi_news'
})
multi_news = multi_news.filter(lambda x: len(x['document']) >=2 )
multi_news = multi_news.map(lambda x: {
    "document": maybe_chunking(x['document'], n=1024)
})
print(multi_news)

n_docs = sum([len(ds) for ds in multi_news['document'][:40000]])
doclength = [len(d.split()) for ds in multi_news['document'][:40000] for d in ds]
sumlength = [len(d.split()) for d in multi_news['summary'][:40000]]

print('====multi_news (train) ====')
print('# Examples:', min(40000, len(multi_news)))
print('# Documents:', n_docs)
print('# Avg doc length:', sum(doclength) // len(doclength))
print('# Avg summary length:', sum(sumlength) // len(sumlength))
print('===========================')

multi_news = load_from_disk("/home/dju/datasets/multi_news")['test']
multi_news = multi_news.map(lambda x: {
    "document": normalize(x['document']), 
    'mds-source': 'multi_news'
})
multi_news = multi_news.filter(lambda x: len(x['document']) >=2 )
multi_news = multi_news.map(lambda x: {
    "document": maybe_chunking(x['document'], n=1024)
})
print(multi_news)

n_docs = sum([len(ds) for ds in multi_news['document'][:5000]])
doclength = [len(d.split()) for ds in multi_news['document'][:5000] for d in ds]
sumlength = [len(d.split()) for d in multi_news['summary'][:5000]]
print('====multi_news (test) ====')
print('# Examples:', min(5000, len(multi_news)))
print('# Documents:', n_docs)
print('# Avg doc length:', sum(doclength) // len(doclength))
print('# Avg summary length:', sum(sumlength) // len(sumlength))
print('===========================')

#
duc04 = load_from_disk("/home/dju/datasets/duc04")['train']
duc04 = duc04.map(lambda x: {
    "document": normalize_list(x['context']),
    "summary": flatten_and_normalize(x['summary']),
    'mds-source': 'duc04'
})
duc04 = duc04.filter(lambda x: len(x['document']) >=2 )
duc04 = duc04.map(lambda x: {
    "document": maybe_chunking(x['document'], n=1024)
})
print(duc04)

n_docs = sum([len(ds) for ds in duc04['document'][:5000]])
doclength = [len(d.split()) for ds in duc04['document'][:5000] for d in ds]
sumlength = [len(d.split()) for d in duc04['summary'][:5000]]
print('====DUC2004 (test) ====')
print('# Examples:', min(50, len(duc04)))
print('# Documents:', n_docs)
print('# Avg doc length:', sum(doclength) // len(doclength))
print('# Avg summary length:', sum(sumlength) // len(sumlength))
print('===========================')
