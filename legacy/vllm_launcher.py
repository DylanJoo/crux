import argparse
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams
# cleaning
import contextlib
import gc
import torch
from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel
)
# from vllm_api import PROMPT # this is the prompt in Andrew's example
PROMPT  = "Hello world"

def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()

async def iterate_over_output_for_one_prompt(output_iterator: AsyncStream) -> str:
    last_text = ""
    prompt = "???"

    async for output in output_iterator:
        prompt = output.prompt
        last_text = output.outputs[0].text
        # print(output.request_id)

    return last_text

async def generate(
    engine: AsyncLLMEngine, 
    request_ids: list[str], 
    prompts: list[str], 
    sampling_params: SamplingParams, 
    **kwargs
) -> list[str]:

    output_iterators = [
        await engine.add_request(request_ids[i], prompt, sampling_params)\
                for i, prompt in enumerate(prompts)
    ]
    outputs = await asyncio.gather(*[iterate_over_output_for_one_prompt(output_iterator)
                                     for output_iterator in output_iterators])
    return list(outputs)

async def main():
    args = parse_args()
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs.from_cli_args(args))

    sampling_params = SamplingParams(
            max_tokens=512, 
            temperature=0.7, 
            top_p=0.95,
            ignore_eos=True,
            skip_special_tokens=False
    )

    ## customize here: load data and prepare your prompts
    prompts = [f"Tell a 500 word story about Amsterdam with the number of {i}." for i in range(100)]
    # prompts = [PROMPT for i in range(100)]

    ## generate and wait
    outputs = await generate(engine, [str(i) for i in range(100)], prompts, sampling_params)

    for p, o in zip(prompts, outputs):
        print("\n# Prompt:", p, "\n-->", o)

    cleanup()
    print('done')

if __name__ == "__main__":
    asyncio.run(main())
