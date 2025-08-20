import math
import argparse
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
import uuid
import re
from typing import List
import logging

## Verbose filtered for HTTP200
class vllmFilter(logging.Filter):
    def filter(self, record):
        return ' request' not in record.getMessage()
logger = logging.getLogger()  # root logger
logger.addFilter(vllmFilter())

class LLM:

    def __init__(
        self,
        model_name_or_path: str = 'meta-llama/Llama-3.2-1B-Instruct',
        temperature=0.0,
        top_p=1.0,
        logprobs=None,
        max_tokens=128,
        dtype='bfloat16',
        num_gpus=1, 
        max_model_len=8196,
        **kwargs
    ):
        model_name_or_path = kwargs.pop('model', None) or model_name_or_path
        args = AsyncEngineArgs(
            model=model_name_or_path,
            dtype=dtype,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=num_gpus,
            pipeline_parallel_size=1,
            max_model_len=max_model_len,
        )
        self.model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs.from_cli_args(args))

        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            logprobs=logprobs,
            skip_special_tokens=False,
            min_tokens=1,
            max_tokens=max_tokens,
        )
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # there is no actively running loop
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    async def _iterate_over_output(self, output_iterator: AsyncStream) -> str:
        async for output in output_iterator:
            output = last_text = output.outputs[0].text
        return output

    def generate(self, prompts, binary_probs=False, dist_logp=False):
        if isinstance(prompts, str):
            prompts = [prompts]

        return self.loop.run_until_complete(self._agenerate(prompts))

    async def _agenerate(self, prompts, **kwargs):
        request_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]

        # Add requests to the engine
        output_iterators = [
            await self.model.add_request(request_id, prompt, self.sampling_params)
            for request_id, prompt in zip(request_ids, prompts)
        ]

        # Gather all the outputs
        outputs = await asyncio.gather(*[
            self._iterate_over_output(output_iterator)
            for output_iterator in output_iterators
        ])
        return list(outputs)
