import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import json
import re
import os
import string
import time

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-4}GB' # original is -6
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def load_model(
    model_name_or_path, 
    dtype=torch.float16, 
    load_mode=None, 
    flash_attention_2=False,
):
    # Load a huggingface model and tokenizer
    # dtype: torch.float16 or torch.bfloat16
    # load_mode: whether to use int8/int4 quantization
    # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

    # Load the FP16 model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    logger.warn(f"Use LLM.{load_mode}")
    start_time = time.time()

    if flash_attention_2:
        model_kwargs = {
            'torch_dtype': torch.bfloat16,
            'attn_implementation': "flash_attention_2"
        }
    else:
        model_kwargs = {'torch_dtype': dtype}

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        load_in_8bit=(load_mode == '8bit'),
        load_in_4bit=(load_mode == '4bit'),
        **model_kwargs
    )
        # max_memory=get_max_memory(),
        # attn_implementation="sdpa",
    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    # Fix OPT bos token problem in HF
    if "opt" in model_name_or_path:
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"

    return model, tokenizer

# def make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=None):
#     # For doc prompt:
#     # - {ID}: doc id (starting from 1)
#     # - {T}: title
#     # - {P}: text
#     # use_shorter: None, "summary", or "extraction"
#
#     text = doc['text']
#     if use_shorter is not None:
#         text = doc[use_shorter]
#     return doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id+1))

# def make_demo(item, prompt, ndoc=None, doc_prompt=None, instruction=None, use_shorter=None, test=False):
#     # For demo prompt
#     # - {INST}: the instruction
#     # - {D}: the documents
#     # - {Q}: the question
#     # - {A}: the answers
#     # ndoc: number of documents to put in context
#     # use_shorter: None, "summary", or "extraction"
#
#     prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question']) # NECESSARY
#     if "{D}" in prompt:
#         if ndoc == 0:
#             prompt = prompt.replace("{D}\n", "") # if there is no doc we also delete the empty line
#         else:
#             doc_list = get_shorter_text(item, item["docs"], ndoc, use_shorter) if use_shorter is not None else item["docs"][:ndoc]
#             text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
#             prompt = prompt.replace("{D}", text)
#
#     if not test:
#         answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
#         prompt = prompt.replace("{A}", "").rstrip() + answer
#     else:
#         prompt = prompt.replace("{A}", "").rstrip() # remove any space or \n
#
#     return prompt
