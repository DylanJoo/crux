from vllm import LLM as v_llm 
from vllm import SamplingParams

class LLM:

    def __init__(
        self,
        model_name_or_path: str = 'meta-llama/Llama-3.2-1B-Instruct',
        temperature=0.0,
        top_p=1.0,
        logprobs=None,
        max_tokens=128,
        dtype='half',
        gpu_memory_utilization=0.9,
        num_gpus=1, 
        max_model_len=10240,
        **kwargs
    ):
        model_name_or_path = kwargs.pop('model', None) or model_name_or_path
        self.model = v_llm(
            model_name_or_path,
            dtype='bfloat16',
            enforce_eager=True,
            tensor_parallel_size=num_gpus or 1
        )
        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            skip_special_tokens=False
        )
        self.max_tokens = max_tokens

    def generate(self, x, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]

        self.sampling_params.max_tokens = self.max_tokens
        self.sampling_params.min_tokens = kwargs.pop('min_tokens', 32)

        output = self.model.generate(x, self.sampling_params)
        if len(output) == 1:
            return [output[0].outputs[0].text]
        else:
            return [o.outputs[0].text for o in output] 

