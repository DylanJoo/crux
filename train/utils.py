import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import json
import re
import os
import string
import time

def update_tokenizer(tokenizer, max_n_contexts=10):
    # for i in range(1, max_n_contexts+1):
    #     tokenizer.add_tokens(f"[{i}]")

    tokenizer.add_special_tokens({"additional_special_tokens": ["<cls>"]})
    return tokenizer

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-4}GB' # original is -6
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

def load_model(model_name_or_path, model_class='causualLM', dtype=torch.float16, load_mode=None):

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    from models import FiDT5
    MODEL_CLASS = {"fid": FiDT5, "seq2seq": AutoModelForSeq2SeqLM}[model_class]

    logger.info(f"Loading {model_name_or_path} ({model_class}) in {dtype}...")
    logger.warn(f"Use generator.{load_mode}")
    start_time = time.time()

    ## model_args
    if model_class == 'causualLM': # one model may larger than a gpu
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map='cuda',
            torch_dtype=dtype,
            load_in_4bit=True,
            max_memory=get_max_memory(),
        )
    else:
        model = MODEL_CLASS.from_pretrained(model_name_or_path).to('cuda')

    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    return model, tokenizer

