import torch
import random
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import (
    PaddingStrategy, 
    PreTrainedTokenizerBase
)

import torch
from transformers.utils import is_torch_fx_proxy

def _shift_right(
    input_ids,
    decoder_start_token_id=0,
    pad_token_id=0
):
    # shift inputs to the right
    if is_torch_fx_proxy(input_ids):
        shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
        shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
    else:
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

@dataclass
class Standard:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    max_src_length: Optional[int] = 1024
    max_tgt_length: Optional[int] = 512
    max_num_contexts: Optional[int] = None
    num_contexts: Optional[int] = None
    num_distractor_docs: Optional[int] = None
    num_redundant_docs: Optional[int] = None
    n_psgs_per_doc: Optional[int] = 1
    shuffle: Optional[bool] = True
    src_template = "Summarize context based on the topic. Write `unrelated` if context is irrelevant and write `redundant` if the information has been summarized in the former contexts. Topic: {} Context: [{}] {} [/{}]"
    tgt_template = "[{}] {}"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # prepare source and target 
        src, tgt = [], []
        for example in features:

            # compressed ctx with distractors
            doc_ctxs = example['doc_ctxs']  
            doc_ctxs_d = example['distract_ctxs'][:self.num_distractor_docs]
            src = (doc_ctxs[:self.max_num_contexts] + doc_ctxs_d)

            comp_ctxs = [" ".join(psg_list[:self.n_psgs_per_doc]) for psg_list in example['comp_ctxs']]
            comp_ctxs_d = ["unrelated."] * len(doc_ctxs_d)
            tgt = (comp_ctxs[:self.max_num_contexts] + comp_ctxs_d)

            # redundant ctxs
            if self.num_redundant_docs >= 1:
                doc_ctxs_r = example['redundant_ctxs'][:self.num_redundant_docs]
                # comp_ctxs_r = ["redundant."] * len(doc_ctxs_r)
                for j, doc_ctx in enumerate(doc_ctxs_r):
                    src += [doc_ctx]
                    tgt += ["redundant."]

            assert len(src) == len(tgt), 'inconsistent length.'

            n = len(src)
            if self.shuffle:
                random_idx = random.sample(range(n), n)
            else:
                random_idx = list(range(n))

            src_, tgt_ = [], ""
            topic = example['topic']

            for i, idx in enumerate(random_idx):
                doc_ctx = src[idx]
                comp_ctx = tgt[idx]
                src_ += [self.src_template.format(topic, i+1, doc_ctx, i+1)]
                tgt_ += self.tgt_template.format(i+1, comp_ctx)

        # tokenize
        inputs = self.tokenizer(
            src_,
            max_length=self.max_src_length-1,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        outputs = self.tokenizer(
            tgt_,
            max_length=self.max_tgt_length-1,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        target_mask = outputs['attention_mask'].bool()
        inputs['labels'] = outputs['input_ids'].masked_fill(~target_mask, -100)

        # define number of contexts
        if self.num_contexts is None:
            N = inputs['input_ids'].size(0)
        else:
            N = self.num_contexts

        # reshape
        inputs['input_ids'] = inputs['input_ids'].view(
            -1, N, inputs['input_ids'].size(-1)
        )
        inputs['attention_mask'] = inputs['attention_mask'].view(
            -1, N*inputs['attention_mask'].size(-1)
        )
        return inputs

@dataclass
class StandardWithPrefix:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    max_src_length: Optional[int] = 1024
    max_tgt_length: Optional[int] = 256
    n_contexts: Optional[int] = None
    shuffle: Optional[bool] = True
    src_template = "topic: {} context: [{}] {} [/{}]"
    tgt_template = "{}"
    prefix_template = "[{}] "

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # prepare source and target 
        src, tgt, prefix = [], [], []
        for example in features:
            n = len(example['doc_ctxs'])
            if self.shuffle:
                random_idx = random.sample(range(n), n)[:self.n_contexts]
            else:
                random_idx = list(range(n))[:self.n_contexts]
            src_ = [] 
            topic = example['question']

            for i, idx in enumerate(random_idx):
                src_ += [self.src_template.format(topic, i+1, example['doc_ctxs'][idx], i+1)]

            true_random_idx = [(i, idx) for i, idx in enumerate(random_idx) if example['labels'][idx] == 1]
            false_random_idx = [(i, idx) for i, idx in enumerate(random_idx) if example['labels'][idx] == 0]

            for i, idx in true_random_idx[:1]:
                prefix_ = self.prefix_template.format(i+1)
                prefix.append(prefix_)
                tgt_ = (prefix_ + self.tgt_template.format(example['comp_ctxs'][idx][0]))
                tgt.append(tgt_)

            for i, idx in false_random_idx[:1]:
                prefix_ = self.prefix_template.format(i+1)
                prefix.append(prefix_)
                tgt_ = (prefix_ + "unrelated.")
                tgt.append(tgt_)

            src += src_ * len(tgt)

        # tokenize
        inputs = self.tokenizer(
            src,
            max_length=self.max_src_length-1,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        outputs = self.tokenizer(
            tgt,
            max_length=self.max_tgt_length-1,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        target_mask = outputs['attention_mask'].bool()
        # inputs['decoder_input_ids'] = _shift_right(outputs['input_ids'])
        inputs['labels'] = outputs['input_ids'].masked_fill(~target_mask, -100)

        # prefix_length = [len(l) for l in self.tokenizer(prefix, add_special_tokens=False).input_ids]
        # for i, l in enumerate(prefix_length):
        #     inputs['labels'][i, :l] = -100 # maske the prefix

        # define batch size (and n_context)
        BS = len(tgt)

        # reshape
        inputs['input_ids'] = inputs['input_ids'].view(
            BS, -1, inputs['input_ids'].size(-1)
        )
        inputs['attention_mask'] = inputs['attention_mask'].view(
            BS, -1
        )
        return inputs

