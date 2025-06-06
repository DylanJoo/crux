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
from vllm_api import PROMPT

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
    # prompts = [f"Tell a 500 word story about Amsterdam with the number of {i}." for i in range(100)]
    prompts = [PROMPT for i in range(100)]

    ## generate and wait
    outputs = await generate(engine, [str(i) for i in range(100)], prompts, sampling_params)

    for p, o in zip(prompts, outputs):
        print("\n# Prompt:", p, "\n-->", o)

    cleanup()
    print('done')

# Detail arguments: 
# usage: vllm_launcher.py [-h] [--model MODEL]
#                        [--task {auto,generate,embedding,embed,classify,score,reward}]
#                        [--tokenizer TOKENIZER] [--skip-tokenizer-init]
#                        [--revision REVISION] [--code-revision CODE_REVISION]
#                        [--tokenizer-revision TOKENIZER_REVISION]
#                        [--tokenizer-mode {auto,slow,mistral}]
#                        [--trust-remote-code]
#                        [--allowed-local-media-path ALLOWED_LOCAL_MEDIA_PATH]
#                        [--download-dir DOWNLOAD_DIR]
#                        [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer,sharded_state,gguf,bitsandbytes,mistral,runai_streamer}]
#                        [--config-format {auto,hf,mistral}]
#                        [--dtype {auto,half,float16,bfloat16,float,float32}]
#                        [--kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}]
#                        [--max-model-len MAX_MODEL_LEN]
#                        [--guided-decoding-backend {outlines,lm-format-enforcer,xgrammar}]
#                        [--logits-processor-pattern LOGITS_PROCESSOR_PATTERN]
#                        [--distributed-executor-backend {ray,mp}]
#                        [--worker-use-ray]
#                        [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE]
#                        [--tensor-parallel-size TENSOR_PARALLEL_SIZE]
#                        [--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS]
#                        [--ray-workers-use-nsight]
#                        [--block-size {8,16,32,64,128}]
#                        [--enable-prefix-caching | --no-enable-prefix-caching]
#                        [--disable-sliding-window] [--use-v2-block-manager]
#                        [--num-lookahead-slots NUM_LOOKAHEAD_SLOTS]
#                        [--seed SEED] [--swap-space SWAP_SPACE]
#                        [--cpu-offload-gb CPU_OFFLOAD_GB]
#                        [--gpu-memory-utilization GPU_MEMORY_UTILIZATION]
#                        [--num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE]
#                        [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS]
#                        [--max-num-seqs MAX_NUM_SEQS]
#                        [--max-logprobs MAX_LOGPROBS] [--disable-log-stats]
#                        [--quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,None}]
#                        [--rope-scaling ROPE_SCALING] [--rope-theta ROPE_THETA]
#                        [--hf-overrides HF_OVERRIDES] [--enforce-eager]
#                        [--max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE]
#                        [--disable-custom-all-reduce]
#                        [--tokenizer-pool-size TOKENIZER_POOL_SIZE]
#                        [--tokenizer-pool-type TOKENIZER_POOL_TYPE]
#                        [--tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG]
#                        [--limit-mm-per-prompt LIMIT_MM_PER_PROMPT]
#                        [--mm-processor-kwargs MM_PROCESSOR_KWARGS]
#                        [--disable-mm-preprocessor-cache] [--enable-lora]
#                        [--enable-lora-bias] [--max-loras MAX_LORAS]
#                        [--max-lora-rank MAX_LORA_RANK]
#                        [--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE]
#                        [--lora-dtype {auto,float16,bfloat16}]
#                        [--long-lora-scaling-factors LONG_LORA_SCALING_FACTORS]
#                        [--max-cpu-loras MAX_CPU_LORAS] [--fully-sharded-loras]
#                        [--enable-prompt-adapter]
#                        [--max-prompt-adapters MAX_PROMPT_ADAPTERS]
#                        [--max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN]
#                        [--device {auto,cuda,neuron,cpu,openvino,tpu,xpu,hpu}]
#                        [--num-scheduler-steps NUM_SCHEDULER_STEPS]
#                        [--multi-step-stream-outputs [MULTI_STEP_STREAM_OUTPUTS]]
#                        [--scheduler-delay-factor SCHEDULER_DELAY_FACTOR]
#                        [--enable-chunked-prefill [ENABLE_CHUNKED_PREFILL]]
#                        [--speculative-model SPECULATIVE_MODEL]
#                        [--speculative-model-quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,None}]
#                        [--num-speculative-tokens NUM_SPECULATIVE_TOKENS]
#                        [--speculative-disable-mqa-scorer]
#                        [--speculative-draft-tensor-parallel-size SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE]
#                        [--speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN]
#                        [--speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE]
#                        [--ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX]
#                        [--ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN]
#                        [--spec-decoding-acceptance-method {rejection_sampler,typical_acceptance_sampler}]
#                        [--typical-acceptance-sampler-posterior-threshold TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD]
#                        [--typical-acceptance-sampler-posterior-alpha TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA]
#                        [--disable-logprobs-during-spec-decoding [DISABLE_LOGPROBS_DURING_SPEC_DECODING]]
#                        [--model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG]
#                        [--ignore-patterns IGNORE_PATTERNS]
#                        [--preemption-mode PREEMPTION_MODE]
#                        [--served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]]
#                        [--qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH]
#                        [--otlp-traces-endpoint OTLP_TRACES_ENDPOINT]
#                        [--collect-detailed-traces COLLECT_DETAILED_TRACES]
#                        [--disable-async-output-proc]
#                        [--scheduling-policy {fcfs,priority}]
#                        [--override-neuron-config OVERRIDE_NEURON_CONFIG]
#                        [--override-pooler-config OVERRIDE_POOLER_CONFIG]
#                        [--compilation-config COMPILATION_CONFIG]
#                        [--kv-transfer-config KV_TRANSFER_CONFIG]
#                        [--worker-cls WORKER_CLS]
#                        [--generation-config GENERATION_CONFIG]
#                        [--calculate-kv-scales] [--disable-log-requests]
#
# options:
#   -h, --help            show this help message and exit
#   --model MODEL         Name or path of the huggingface model to use.
#   --task {auto,generate,embedding,embed,classify,score,reward}
#                         The task to use the model for. Each vLLM instance only
#                         supports one task, even if the same model can be used
#                         for multiple tasks. When the model only supports one
#                         task, "auto" can be used to select it; otherwise, you
#                         must specify explicitly which task to use.
#   --tokenizer TOKENIZER
#                         Name or path of the huggingface tokenizer to use. If
#                         unspecified, model name or path will be used.
#   --skip-tokenizer-init
#                         Skip initialization of tokenizer and detokenizer
#   --revision REVISION   The specific model version to use. It can be a branch
#                         name, a tag name, or a commit id. If unspecified, will
#                         use the default version.
#   --code-revision CODE_REVISION
#                         The specific revision to use for the model code on
#                         Hugging Face Hub. It can be a branch name, a tag name,
#                         or a commit id. If unspecified, will use the default
#                         version.
#   --tokenizer-revision TOKENIZER_REVISION
#                         Revision of the huggingface tokenizer to use. It can
#                         be a branch name, a tag name, or a commit id. If
#                         unspecified, will use the default version.
#   --tokenizer-mode {auto,slow,mistral}
#                         The tokenizer mode. * "auto" will use the fast
#                         tokenizer if available. * "slow" will always use the
#                         slow tokenizer. * "mistral" will always use the
#                         `mistral_common` tokenizer.
#   --trust-remote-code   Trust remote code from huggingface.
#   --allowed-local-media-path ALLOWED_LOCAL_MEDIA_PATH
#                         Allowing API requests to read local images or videos
#                         from directories specified by the server file system.
#                         This is a security risk. Should only be enabled in
#                         trusted environments.
#   --download-dir DOWNLOAD_DIR
#                         Directory to download and load the weights, default to
#                         the default cache dir of huggingface.
#   --load-format {auto,pt,safetensors,npcache,dummy,tensorizer,sharded_state,gguf,bitsandbytes,mistral,runai_streamer}
#                         The format of the model weights to load. * "auto" will
#                         try to load the weights in the safetensors format and
#                         fall back to the pytorch bin format if safetensors
#                         format is not available. * "pt" will load the weights
#                         in the pytorch bin format. * "safetensors" will load
#                         the weights in the safetensors format. * "npcache"
#                         will load the weights in pytorch format and store a
#                         numpy cache to speed up the loading. * "dummy" will
#                         initialize the weights with random values, which is
#                         mainly for profiling. * "tensorizer" will load the
#                         weights using tensorizer from CoreWeave. See the
#                         Tensorize vLLM Model script in the Examples section
#                         for more information. * "runai_streamer" will load the
#                         Safetensors weights using Run:aiModel Streamer *
#                         "bitsandbytes" will load the weights using
#                         bitsandbytes quantization.
#   --config-format {auto,hf,mistral}
#                         The format of the model config to load. * "auto" will
#                         try to load the config in hf format if available else
#                         it will try to load in mistral format
#   --dtype {auto,half,float16,bfloat16,float,float32}
#                         Data type for model weights and activations. * "auto"
#                         will use FP16 precision for FP32 and FP16 models, and
#                         BF16 precision for BF16 models. * "half" for FP16.
#                         Recommended for AWQ quantization. * "float16" is the
#                         same as "half". * "bfloat16" for a balance between
#                         precision and range. * "float" is shorthand for FP32
#                         precision. * "float32" for FP32 precision.
#   --kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}
#                         Data type for kv cache storage. If "auto", will use
#                         model data type. CUDA 11.8+ supports fp8 (=fp8_e4m3)
#                         and fp8_e5m2. ROCm (AMD GPU) supports fp8 (=fp8_e4m3)
#   --max-model-len MAX_MODEL_LEN
#                         Model context length. If unspecified, will be
#                         automatically derived from the model config.
#   --guided-decoding-backend {outlines,lm-format-enforcer,xgrammar}
#                         Which engine will be used for guided decoding (JSON
#                         schema / regex etc) by default. Currently support
#                         https://github.com/outlines-dev/outlines,
#                         https://github.com/mlc-ai/xgrammar, and
#                         https://github.com/noamgat/lm-format-enforcer. Can be
#                         overridden per request via guided_decoding_backend
#                         parameter.
#   --logits-processor-pattern LOGITS_PROCESSOR_PATTERN
#                         Optional regex pattern specifying valid logits
#                         processor qualified names that can be passed with the
#                         `logits_processors` extra completion argument.
#                         Defaults to None, which allows no processors.
#   --distributed-executor-backend {ray,mp}
#                         Backend to use for distributed model workers, either
#                         "ray" or "mp" (multiprocessing). If the product of
#                         pipeline_parallel_size and tensor_parallel_size is
#                         less than or equal to the number of GPUs available,
#                         "mp" will be used to keep processing on a single host.
#                         Otherwise, this will default to "ray" if Ray is
#                         installed and fail otherwise. Note that tpu and hpu
#                         only support Ray for distributed inference.
#   --worker-use-ray      Deprecated, use --distributed-executor-backend=ray.
#   --pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE
#                         Number of pipeline stages.
#   --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
#                         Number of tensor parallel replicas.
#   --max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS
#                         Load model sequentially in multiple batches, to avoid
#                         RAM OOM when using tensor parallel and large models.
#   --ray-workers-use-nsight
#                         If specified, use nsight to profile Ray workers.
#   --block-size {8,16,32,64,128}
#                         Token block size for contiguous chunks of tokens. This
#                         is ignored on neuron devices and set to max-model-len.
#                         On CUDA devices, only block sizes up to 32 are
#                         supported. On HPU devices, block size defaults to 128.
#   --enable-prefix-caching, --no-enable-prefix-caching
#                         Enables automatic prefix caching. Use --no-enable-
#                         prefix-caching to disable explicitly.
#   --disable-sliding-window
#                         Disables sliding window, capping to sliding window
#                         size
#   --use-v2-block-manager
#                         [DEPRECATED] block manager v1 has been removed and
#                         SelfAttnBlockSpaceManager (i.e. block manager v2) is
#                         now the default. Setting this flag to True or False
#                         has no effect on vLLM behavior.
#   --num-lookahead-slots NUM_LOOKAHEAD_SLOTS
#                         Experimental scheduling config necessary for
#                         speculative decoding. This will be replaced by
#                         speculative config in the future; it is present to
#                         enable correctness tests until then.
#   --seed SEED           Random seed for operations.
#   --swap-space SWAP_SPACE
#                         CPU swap space size (GiB) per GPU.
#   --cpu-offload-gb CPU_OFFLOAD_GB
#                         The space in GiB to offload to CPU, per GPU. Default
#                         is 0, which means no offloading. Intuitively, this
#                         argument can be seen as a virtual way to increase the
#                         GPU memory size. For example, if you have one 24 GB
#                         GPU and set this to 10, virtually you can think of it
#                         as a 34 GB GPU. Then you can load a 13B model with
#                         BF16 weight, which requires at least 26GB GPU memory.
#                         Note that this requires fast CPU-GPU interconnect, as
#                         part of the model is loaded from CPU memory to GPU
#                         memory on the fly in each model forward pass.
#   --gpu-memory-utilization GPU_MEMORY_UTILIZATION
#                         The fraction of GPU memory to be used for the model
#                         executor, which can range from 0 to 1. For example, a
#                         value of 0.5 would imply 50% GPU memory utilization.
#                         If unspecified, will use the default value of 0.9.
#                         This is a per-instance limit, and only applies to the
#                         current vLLM instance.It does not matter if you have
#                         another vLLM instance running on the same GPU. For
#                         example, if you have two vLLM instances running on the
#                         same GPU, you can set the GPU memory utilization to
#                         0.5 for each instance.
#   --num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE
#                         If specified, ignore GPU profiling result and use this
#                         number of GPU blocks. Used for testing preemption.
#   --max-num-batched-tokens MAX_NUM_BATCHED_TOKENS
#                         Maximum number of batched tokens per iteration.
#   --max-num-seqs MAX_NUM_SEQS
#                         Maximum number of sequences per iteration.
#   --max-logprobs MAX_LOGPROBS
#                         Max number of log probs to return logprobs is
#                         specified in SamplingParams.
#   --disable-log-stats   Disable logging statistics.
#   --quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,None}, -q {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,None}
#                         Method used to quantize the weights. If None, we first
#                         check the `quantization_config` attribute in the model
#                         config file. If that is None, we assume the model
#                         weights are not quantized and use `dtype` to determine
#                         the data type of the weights.
#   --rope-scaling ROPE_SCALING
#                         RoPE scaling configuration in JSON format. For
#                         example, {"rope_type":"dynamic","factor":2.0}
#   --rope-theta ROPE_THETA
#                         RoPE theta. Use with `rope_scaling`. In some cases,
#                         changing the RoPE theta improves the performance of
#                         the scaled model.
#   --hf-overrides HF_OVERRIDES
#                         Extra arguments for the HuggingFace config. This
#                         should be a JSON string that will be parsed into a
#                         dictionary.
#   --enforce-eager       Always use eager-mode PyTorch. If False, will use
#                         eager mode and CUDA graph in hybrid for maximal
#                         performance and flexibility.
#   --max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE
#                         Maximum sequence length covered by CUDA graphs. When a
#                         sequence has context length larger than this, we fall
#                         back to eager mode. Additionally for encoder-decoder
#                         models, if the sequence length of the encoder input is
#                         larger than this, we fall back to the eager mode.
#   --disable-custom-all-reduce
#                         See ParallelConfig.
#   --tokenizer-pool-size TOKENIZER_POOL_SIZE
#                         Size of tokenizer pool to use for asynchronous
#                         tokenization. If 0, will use synchronous tokenization.
#   --tokenizer-pool-type TOKENIZER_POOL_TYPE
#                         Type of tokenizer pool to use for asynchronous
#                         tokenization. Ignored if tokenizer_pool_size is 0.
#   --tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG
#                         Extra config for tokenizer pool. This should be a JSON
#                         string that will be parsed into a dictionary. Ignored
#                         if tokenizer_pool_size is 0.
#   --limit-mm-per-prompt LIMIT_MM_PER_PROMPT
#                         For each multimodal plugin, limit how many input
#                         instances to allow for each prompt. Expects a comma-
#                         separated list of items, e.g.: `image=16,video=2`
#                         allows a maximum of 16 images and 2 videos per prompt.
#                         Defaults to 1 for each modality.
#   --mm-processor-kwargs MM_PROCESSOR_KWARGS
#                         Overrides for the multimodal input mapping/processing,
#                         e.g., image processor. For example: {"num_crops": 4}.
#   --disable-mm-preprocessor-cache
#                         If true, then disables caching of the multi-modal
#                         preprocessor/mapper. (not recommended)
#   --enable-lora         If True, enable handling of LoRA adapters.
#   --enable-lora-bias    If True, enable bias for LoRA adapters.
#   --max-loras MAX_LORAS
#                         Max number of LoRAs in a single batch.
#   --max-lora-rank MAX_LORA_RANK
#                         Max LoRA rank.
#   --lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE
#                         Maximum size of extra vocabulary that can be present
#                         in a LoRA adapter (added to the base model
#                         vocabulary).
#   --lora-dtype {auto,float16,bfloat16}
#                         Data type for LoRA. If auto, will default to base
#                         model dtype.
#   --long-lora-scaling-factors LONG_LORA_SCALING_FACTORS
#                         Specify multiple scaling factors (which can be
#                         different from base model scaling factor - see eg.
#                         Long LoRA) to allow for multiple LoRA adapters trained
#                         with those scaling factors to be used at the same
#                         time. If not specified, only adapters trained with the
#                         base model scaling factor are allowed.
#   --max-cpu-loras MAX_CPU_LORAS
#                         Maximum number of LoRAs to store in CPU memory. Must
#                         be >= than max_loras. Defaults to max_loras.
#   --fully-sharded-loras
#                         By default, only half of the LoRA computation is
#                         sharded with tensor parallelism. Enabling this will
#                         use the fully sharded layers. At high sequence length,
#                         max rank or tensor parallel size, this is likely
#                         faster.
#   --enable-prompt-adapter
#                         If True, enable handling of PromptAdapters.
#   --max-prompt-adapters MAX_PROMPT_ADAPTERS
#                         Max number of PromptAdapters in a batch.
#   --max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN
#                         Max number of PromptAdapters tokens
#   --device {auto,cuda,neuron,cpu,openvino,tpu,xpu,hpu}
#                         Device type for vLLM execution.
#   --num-scheduler-steps NUM_SCHEDULER_STEPS
#                         Maximum number of forward steps per scheduler call.
#   --multi-step-stream-outputs [MULTI_STEP_STREAM_OUTPUTS]
#                         If False, then multi-step will stream outputs at the
#                         end of all steps
#   --scheduler-delay-factor SCHEDULER_DELAY_FACTOR
#                         Apply a delay (of delay factor multiplied by previous
#                         prompt latency) before scheduling next prompt.
#   --enable-chunked-prefill [ENABLE_CHUNKED_PREFILL]
#                         If set, the prefill requests can be chunked based on
#                         the max_num_batched_tokens.
#   --speculative-model SPECULATIVE_MODEL
#                         The name of the draft model to be used in speculative
#                         decoding.
#   --speculative-model-quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,None}
#                         Method used to quantize the weights of speculative
#                         model. If None, we first check the
#                         `quantization_config` attribute in the model config
#                         file. If that is None, we assume the model weights are
#                         not quantized and use `dtype` to determine the data
#                         type of the weights.
#   --num-speculative-tokens NUM_SPECULATIVE_TOKENS
#                         The number of speculative tokens to sample from the
#                         draft model in speculative decoding.
#   --speculative-disable-mqa-scorer
#                         If set to True, the MQA scorer will be disabled in
#                         speculative and fall back to batch expansion
#   --speculative-draft-tensor-parallel-size SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE, -spec-draft-tp SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE
#                         Number of tensor parallel replicas for the draft model
#                         in speculative decoding.
#   --speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN
#                         The maximum sequence length supported by the draft
#                         model. Sequences over this length will skip
#                         speculation.
#   --speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE
#                         Disable speculative decoding for new incoming requests
#                         if the number of enqueue requests is larger than this
#                         value.
#   --ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX
#                         Max size of window for ngram prompt lookup in
#                         speculative decoding.
#   --ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN
#                         Min size of window for ngram prompt lookup in
#                         speculative decoding.
#   --spec-decoding-acceptance-method {rejection_sampler,typical_acceptance_sampler}
#                         Specify the acceptance method to use during draft
#                         token verification in speculative decoding. Two types
#                         of acceptance routines are supported: 1)
#                         RejectionSampler which does not allow changing the
#                         acceptance rate of draft tokens, 2)
#                         TypicalAcceptanceSampler which is configurable,
#                         allowing for a higher acceptance rate at the cost of
#                         lower quality, and vice versa.
#   --typical-acceptance-sampler-posterior-threshold TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD
#                         Set the lower bound threshold for the posterior
#                         probability of a token to be accepted. This threshold
#                         is used by the TypicalAcceptanceSampler to make
#                         sampling decisions during speculative decoding.
#                         Defaults to 0.09
#   --typical-acceptance-sampler-posterior-alpha TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA
#                         A scaling factor for the entropy-based threshold for
#                         token acceptance in the TypicalAcceptanceSampler.
#                         Typically defaults to sqrt of --typical-acceptance-
#                         sampler-posterior-threshold i.e. 0.3
#   --disable-logprobs-during-spec-decoding [DISABLE_LOGPROBS_DURING_SPEC_DECODING]
#                         If set to True, token log probabilities are not
#                         returned during speculative decoding. If set to False,
#                         log probabilities are returned according to the
#                         settings in SamplingParams. If not specified, it
#                         defaults to True. Disabling log probabilities during
#                         speculative decoding reduces latency by skipping
#                         logprob calculation in proposal sampling, target
#                         sampling, and after accepted tokens are determined.
#   --model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG
#                         Extra config for model loader. This will be passed to
#                         the model loader corresponding to the chosen
#                         load_format. This should be a JSON string that will be
#                         parsed into a dictionary.
#   --ignore-patterns IGNORE_PATTERNS
#                         The pattern(s) to ignore when loading the
#                         model.Default to `original/**/*` to avoid repeated
#                         loading of llama's checkpoints.
#   --preemption-mode PREEMPTION_MODE
#                         If 'recompute', the engine performs preemption by
#                         recomputing; If 'swap', the engine performs preemption
#                         by block swapping.
#   --served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]
#                         The model name(s) used in the API. If multiple names
#                         are provided, the server will respond to any of the
#                         provided names. The model name in the model field of a
#                         response will be the first name in this list. If not
#                         specified, the model name will be the same as the
#                         `--model` argument. Noted that this name(s) will also
#                         be used in `model_name` tag content of prometheus
#                         metrics, if multiple names provided, metrics tag will
#                         take the first one.
#   --qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH
#                         Name or path of the QLoRA adapter.
#   --otlp-traces-endpoint OTLP_TRACES_ENDPOINT
#                         Target URL to which OpenTelemetry traces will be sent.
#   --collect-detailed-traces COLLECT_DETAILED_TRACES
#                         Valid choices are model,worker,all. It makes sense to
#                         set this only if --otlp-traces-endpoint is set. If
#                         set, it will collect detailed traces for the specified
#                         modules. This involves use of possibly costly and or
#                         blocking operations and hence might have a performance
#                         impact.
#   --disable-async-output-proc
#                         Disable async output processing. This may result in
#                         lower performance.
#   --scheduling-policy {fcfs,priority}
#                         The scheduling policy to use. "fcfs" (first come first
#                         served, i.e. requests are handled in order of arrival;
#                         default) or "priority" (requests are handled based on
#                         given priority (lower value means earlier handling)
#                         and time of arrival deciding any ties).
#   --override-neuron-config OVERRIDE_NEURON_CONFIG
#                         Override or set neuron device configuration. e.g.
#                         {"cast_logits_dtype": "bloat16"}.'
#   --override-pooler-config OVERRIDE_POOLER_CONFIG
#                         Override or set the pooling method for pooling models.
#                         e.g. {"pooling_type": "mean", "normalize": false}.'
#   --compilation-config COMPILATION_CONFIG, -O COMPILATION_CONFIG
#                         torch.compile configuration for the model.When it is a
#                         number (0, 1, 2, 3), it will be interpreted as the
#                         optimization level. NOTE: level 0 is the default level
#                         without any optimization. level 1 and 2 are for
#                         internal testing only. level 3 is the recommended
#                         level for production. To specify the full compilation
#                         config, use a JSON string. Following the convention of
#                         traditional compilers, using -O without space is also
#                         supported. -O3 is equivalent to -O 3.
#   --kv-transfer-config KV_TRANSFER_CONFIG
#                         The configurations for distributed KV cache transfer.
#                         Should be a JSON string.
#   --worker-cls WORKER_CLS
#                         The worker class to use for distributed execution.
#   --generation-config GENERATION_CONFIG
#                         The folder path to the generation config. Defaults to
#                         None, will use the default generation config in vLLM.
#                         If set to 'auto', the generation config will be
#                         automatically loaded from model. If set to a folder
#                         path, the generation config will be loaded from the
#                         specified folder path.
#   --calculate-kv-scales
#                         This enables dynamic calculation of k_scale and
#                         v_scale when kv-cache-dtype is fp8. If calculate-kv-
#                         scales is false, the scales will be loaded from the
#                         model checkpoint if available. Otherwise, the scales
#                         will default to 1.0.
#   --disable-log-requests
#                         Disable logging requests.

if __name__ == "__main__":
    asyncio.run(main())
