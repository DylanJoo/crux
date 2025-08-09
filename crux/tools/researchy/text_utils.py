# Utility functions for text processing for researchy-Q 
# TODO: move some common ones to the generic
import re

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def remove_header(question):
    p = re.compile(r"(\s)*-\s") 
    return p.sub("", question).strip()

def normalize_doc(doc):
    doc = re.sub(r'\n+', '\n', doc)
    doc = re.sub(r'\s+', '\n', doc)
    return doc.strip()

def postprocess(sent, tag='p'):
    sent = remove_citations(sent)
    sent = sent.strip().split(f'</{tag}>')[0]
    return sent

def normalize_text(texts):
    texts = unicodedata.normalize('NFKC', texts)
    texts = texts.strip()
    pattern = re.compile(r"\n")
    texts = re.sub(pattern, ' ', texts).strip()
    pattern = re.compile(r"\s+")
    texts = re.sub(pattern, ' ', texts).strip()
    return texts

