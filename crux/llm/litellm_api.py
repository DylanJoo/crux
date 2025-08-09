import os
import math
import asyncio
import openai
from typing import List, Optional, Union
from transformers import AutoTokenizer

## Verbose filtered for HTTP200
import logging
class SuppressHTTP200Filter(logging.Filter):
    def filter(self, record):
        return '"HTTP/1.1 200 OK"' not in record.getMessage()
logger = logging.getLogger()  # root logger
logger.addFilter(SuppressHTTP200Filter())

class LLM:

    def __init__(
        self,
        temperature=0.0,
        top_p=1.0,
        logprobs=None,
        max_tokens=1024, 
        **kwargs
    ):
        self.model = "llama3.3-70b-instruct"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logprobs = logprobs
        self.model_type = 'litellm'

        self.client = openai.OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            base_url='http://10.162.95.158:4000/v1/',
            max_retries=10
        )

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # there is no actively running loop
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-3.3-70B-Instruct',
            use_fast=False
        )

    async def _generate_async(self, prompts: List[str]) -> List[float]:

        # singlge function call of selected token prob
        def _generate(prompt: str) -> float:
            try:
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
            except:
                return "-2"

        # Gather all the outputs
        outputs = await asyncio.gather(*[
            asyncio.to_thread(_generate, prompt) for prompt in prompts
        ])
        return list(outputs)

    # def inference(self, prompts):
    #     if isinstance(prompts, str):
    #         prompts = [prompts]
    #     
    #     return self.loop.run_until_complete(self._generate_async(prompts))

    def generate(
        self, 
        user_prompts: Union[List[str]] = None,
        **kwargs
    ):
        if isinstance(user_prompts, str):
            user_prompts = [user_prompts]

        prompts = [self.tokenizer.apply_chat_template(
            conversation=[
                {"role": "user", "content": prompt}
            ],
            tokenize=False,
            add_generation_prompt=True
        ) for prompt in user_prompts]
        
        return self.loop.run_until_complete(self._generate_async(prompts))
