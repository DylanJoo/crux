from argparse import Namespace
from types import SimpleNamespace
import re

from prompts.neuclir import prompt_subquestion_gen, prompt_complementary_subquestion_gen


def generate_complementary_subtopics(args: Namespace, 
                                     raw_topics: dict, 
                                     ref_subquestions: dict,
                                     k: int = 10,
                                     **kwargs) -> dict:
    """ Generate subquestions complementarily to the reference subquestions
    Args:
        args [Namespace]: args provided to script
        raw_topics [dict]: raw topics as provided in topic file (i.e. example-request.jsonl)
        ref_subquestions [dict]: reference subquestions
        k [int]: number of subquestions to generate
    """

    from llm.litellm_api import LLM
    llm = LLM(
        model=args.model,
        temperature=args.temperature if hasattr(args, 'temperature') else 1.0,
        top_p=args.top_p if hasattr(args, 'top_p') else 0.95,
        max_tokens=512,
    )

    # Generate subtopics with LLM
    subtopics_dict = {}
    prompts = []
    for query in raw_topics:
        prompt = prompt_complementary_subquestion_gen(
            query["background"], query["problem_statement"], 
            query["title"], ref_subquestions[query["request_id"]],
            k=k)
        prompts.append(prompt)

    outputs = llm.inference_chat(prompts)

    for idx, query in enumerate(raw_topics):
        llm_output = outputs[idx]
        pattern = r'<START OF LIST>(.*?)<END OF LIST>'
        match = re.search(pattern, llm_output, flags=re.MULTILINE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
        else:
            extracted = llm_output

        subtopics = extracted.split("\n")
        suptopics = [s for s in subtopics if (s and s not in ["START OF LIST", "END OF LIST"])]

        suptopics = suptopics[:k]
        subtopics_dict[query["request_id"]] = suptopics
    
    return subtopics_dict


def generate_subtopics(args: Namespace, 
                                     topic: dict, 
                                     k: int = 10,
                                     **kwargs) -> dict:
    """ Generate subquestions complementarily to the reference subquestions
    Args:
        args [Namespace]: args provided to script
        raw_topics [dict]: raw topics as provided in topic file (i.e. example-request.jsonl)
        ref_subquestions [dict]: reference subquestions
        k [int]: number of subquestions to generate
    """

    from llm.litellm_api import LLM
    llm = LLM(
        model=args.model,
        temperature=args.temperature if hasattr(args, 'temperature') else 1.0,
        top_p=args.top_p if hasattr(args, 'top_p') else 0.95,
        max_tokens=512,
    )

    # Generate subtopics with LLM
    prompt = prompt_subquestion_gen(
        topic["background"], topic["problem_statement"], 
        topic["title"], k=k)

    outputs = llm.inference_chat([prompt])

    subtopics = extract_llm_generated_subtopics(outputs[0], k)
    
    return subtopics

async def async_generate_subtopics(args: Namespace, 
                                     topic: dict, 
                                     k: int = 10,
                                     **kwargs) -> dict:
    """ Generate subquestions complementarily to the reference subquestions
    Args:
        args [Namespace]: args provided to script
        raw_topics [dict]: raw topics as provided in topic file (i.e. example-request.jsonl)
        ref_subquestions [dict]: reference subquestions
        k [int]: number of subquestions to generate
    """

    from llm.litellm_api import LLM
    llm = LLM(
        model=args.model,
        temperature=args.temperature if hasattr(args, 'temperature') else 1.0,
        top_p=args.top_p if hasattr(args, 'top_p') else 0.95,
        max_tokens=512,
    )

    # Generate subtopics with LLM
    prompt = prompt_subquestion_gen(
        topic["background"], topic["problem_statement"], 
        topic["title"], k=k)

    outputs = await llm.async_inference_chat([prompt])

    subtopics = extract_llm_generated_subtopics(outputs[0], k)
    
    return subtopics

def extract_llm_generated_subtopics(llm_output, k):
    pattern = r'<START OF LIST>(.*?)<END OF LIST>'
    match = re.search(pattern, llm_output, flags=re.MULTILINE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
    else:
        extracted = llm_output

    subtopics = extracted.split("\n")
    suptopics = [s for s in subtopics if (s and s not in ["START OF LIST", "END OF LIST"])]

    suptopics = suptopics[:k]
    #subtopics_dict[query["request_id"]] = suptopics
    return subtopics

