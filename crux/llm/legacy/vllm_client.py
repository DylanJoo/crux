import os
import openai
import math
import asyncio
from openai import OpenAI
from typing import List, Optional, Union
from transformers import AutoTokenizer

class LLM:

    def __init__(
        self,
        model_name_or_path: str = 'meta-llama/Llama-3.2-1B-Instruct',
        temperature=0.0,
        top_p=1.0,
        logprobs=None,
        max_tokens=128,
        dtype='half',
        num_gpus=1, 
        max_model_len=8196,
        **kwargs
    ):
        self.client = OpenAI(
            base_url='http:///127.0.0.1:8000/v1',
            api_key='EMPTY',
        )
        self.model = model_name_or_path
        self.model_type = 'vllm_client'

        self.temperature = temperature
        self.top_p = top_p
        self.logprobs = logprobs
        self.min_tokens = 1
        self.max_tokens = max_tokens

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    async def _generate_async(self, prompts: List[str]) -> List[float]:

        def _generate(prompt: str) -> float:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                logprobs=self.logprobs,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            output_text = response.choices[0].text
            return output_text

        # Gather all the outputs
        outputs = await asyncio.gather(*[
            asyncio.to_thread(_generate, prompt) for prompt in prompts
        ])
        return list(outputs)

    def generate(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        return self.loop.run_until_complete(self._generate_async(prompts))

            # try:
            #     response = self.client.completions.create(
            #         model=self.model,
            #         prompt=prompt,
            #         logprobs=self.logprobs,
            #         temperature=self.temperature,
            #         top_p=self.top_p,
            #         max_tokens=self.max_tokens,
            #     )
            #
            #     output_text = response.choices[0].text
            #     return output_text
            # except:
            #     return "-2"
