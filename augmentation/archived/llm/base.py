import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import argparse
import os
import json
from tqdm import tqdm
import time
import string
import re

from vllm import LLM as v_llm 
from vllm import SamplingParams
from transformers import AutoTokenizer

class vLLM:

    def __init__(self, args):
        self.model = v_llm(
            args.model, 
            enforce_eager=True,
            pipeline_parallel_size=(args.num_gpus or 1)
        )
            # dtype='bfloat16',
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        self.tokenizer.padding_side = "left"
        self.sampling_params = SamplingParams(
            temperature=args.temperature, 
            top_p=args.top_p,
            skip_special_tokens=False
        )

    def generate(self, x, **kwargs):
        self.sampling_params.max_tokens = kwargs.pop('max_tokens', 256)
        self.sampling_params.min_tokens = kwargs.pop('min_tokens', 32)
        output = self.model.generate(x, self.sampling_params)[0].outputs[0].text
        return output
