import os
import json
import argparse
from tqdm import tqdm
from tools.researchy import (
    load_corpus, 
    load_query, 
    load_subtopics,
)
# load_judgements, load_reports,
# load_qrels, load_diversity_qrels
from tools import batch_iterator
from tools import postprocess

def main(args):

    # Data 
    topics = load_topics(args.data.topic_file, args.debug)
    questions = load_questions(args.data.topic_file)
    corpus = load_corpus(args.data.corpus_dir)
    qrels = load_qrels(args.data.qrels_file)
    diversity_qrels = load_diversity_qrels(args.data.qrels_file.replace('qrels', 'div_qrels'))
    judgements = load_judgements(args.data.judgement_file) \
            if args.data.judgement_file is not None else None
    reports = load_reports(args.data.topic_file)

    # Retrieval
    from sim_rag.retrieve.bm25 import search
    output_run = search(
        index=args.data.index_dir,
        k1=args.retrieval.k1, 
        b=args.retrieval.b,
        query=queries,
        batch_size=args.retrieval.batch_size,
        k=args.retrieval.k,
    )

    # Context augmentation
    from augment.base import vanilla
    output_rac = vanilla(
        topics=topics,
        corpus=corpus,
        runs=output_run,
        questions=questions,
        max_k=args.augmentation.max_k if args.augmentation else None,
        qrels=load_qrels(args.data.qrels_file, threshold=3) 
    )

    # Retrieval-augmented context evaluation
    if judgements:
        from evaluation import rac_evaluate
        output_rac_eval = rac_evaluate(
            corpus=corpus,
            qrels=qrels,
            judgements=judgements,
            diversity_qrels=diversity_qrels, 
            rac_data=output_rac,
            n_questions=args.data.n_questions,
            threshold=args.data.threshold,
            runs=output_run,
            tokenizer_name=args.generation.model_name_or_path if args.generation is not None else None
        )
        print(output_rac_eval)

        metrics = ['mean_coverage', 'mean_density', 'Recall', 'MAP', 'nDCG', 'alpha_nDCG']
        values = [str(output_rac_eval[m]) for m in metrics]
        print(" ".join(['RAC-eval'] + metrics))
        print(" ".join(["##" + args.exp] + values))

    # Generation
    # [TODO] Replace this with GPT researcher's report
    # if args.generation is not None:
    #     token_word_limit = {512: "300", 1024: "600", 2048: "1000"}
    #
    #     # old citation generation
    #     PROMPT = \
    #         "Write a passage for the given query. Always use the provided contexts to write the passage (some of the contexts might be irrelevant). " + \
    #         "Cite at least one context in each sentence in the passage. When citing several search results, use [1][2][3]. " + \
    #         "Write the passage within {WORD_LIMIT} words.\n\nQuery: {Q}\nContexts:\n{Ds}\nPassage:\n"
    #
    #     if check_if_ampere:
    #         from generate.llm.vllm_back import LLM
    #     else:
    #         from generate.llm.hf_back import LLM
    #     generator = LLM(
    #         model=args.generation.model_name_or_path, 
    #         temperature=args.generation.temperature,
    #     )
    #     all_prompts, all_qids = {}, []
    #     for qid in topics:
    #         q = output_rac[qid]['topic']
    #         ds = output_rac[qid]['prompt']
    #         all_qids.append(qid)
    #
    #         ### [NOTE] set the oracle length boundary
    #         if args.generation.max_length == -1: 
    #             oracle_word_limit = ( (1+len(reports[qid].split(' ')) // 100) * 100)
    #             max_length = int(oracle_word_limit * 1.5)
    #             word_limit = str(oracle_word_limit)
    #         else:
    #             max_length = args.generation.max_length
    #             word_limit = token_word_limit[max_length]
    #
    #         # print(len(reports[qid].split()), max_length, word_limit)
    #         all_prompts[qid] = PROMPT.replace("{Q}", q).replace("{Ds}", ds).replace("{WORD_LIMIT}", word_limit)
    #
    #     for batch_qid in tqdm(batch_iterator(all_qids, size=args.generation.batch_size), desc="Generating", total=len(topics)//args.generation.batch_size):
    #         responses = generator.generate(x=[all_prompts[qid] for qid in batch_qid], max_tokens=max_length)
    #         for qid, response in zip(batch_qid, responses):
    #             output_rac[qid]['report'] = reports[qid]
    #             output_rac[qid]['response'] = postprocess(response, 'r')
    #
    #     print(cleanup_vllm(generator) if check_if_ampere else "\n")
    #
    #     # output final report as file
    #     os.makedirs(f"results/{args.generation.max_length}", exist_ok=True)
    #     with open(os.path.join(f"results/{args.generation.max_length}", f"{args.exp}.jsonl"), 'w') as f:
    #         for k, data in output_rac.items():
    #             del data['prompt']
    #             del data['context_list']
    #             f.write(json.dumps(data) + '\n')
    #
    #     metrics = ['mean_coverage', 'mean_density', 'Recall', 'MAP', 'nDCG', 'alpha_nDCG']
    #     values = [str(output_rac_eval[m]) for m in metrics]
    #     print(" ".join(['RAC-eval'] + metrics))
    #     print(" ".join(['##' + args.exp] + values))

    # Evaluation
    # [TODO] Final result evaluation
    # if (args.generation is not None) and (args.online_eval is True):
    #     generator = LLM(
    #         model=args.generation.model_name_or_path, 
    #         top_p=1,
    #         temperature=0 if check_if_ampere else 1e-10
    #     )
    #     from evaluation import rag_evaluate
    #     output_rag_eval = rag_evaluate(
    #         generator=generator,
    #         corpus=corpus, 
    #         qrels=qrels, 
    #         judgements=judgements,
    #         rag_data=output_rac,
    #         questions=questions,
    #         threshold=args.data.threshold,
    #         tokenizer_name=args.generation.model_name_or_path,
    #     )
    #     print(output_rac_eval)
    #     print(output_rag_eval)
    #
    #     metrics = ['mean_coverage', 'mean_density', 'Recall', 'MAP', 'nDCG', 'alpha_nDCG']
    #     values = [str(output_rac_eval[m]) for m in metrics]
    #     print(" ".join(['RAC-eval'] + metrics))
    #     print(" ".join(['##' + args.exp] + values))
    #
    #     metrics = ['mean_coverage', 'mean_density']
    #     values =  [str(output_rag_eval['mean_coverage']), str(output_rag_eval['mean_density'])]
    #     print(" ".join(['RAG-eval'] + metrics))
    #     print(" ".join(['##' + args.exp] + values))

if __name__ == "__main__":
    from tools import pretty_print_args
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--default_config", type=str, default=None)
    config_parser.add_argument("--debug", type=int, default=None)
    config_parser.add_argument("--num_gpus", type=int, default=1)
    config_parser.add_argument("--online_eval", action='store_true', default=False)
    config_parser.add_argument("--exp", type=str, default='testing')
    config_args, remaining_argv = config_parser.parse_known_args()
    
    yaml_config = load_yaml_config(config_args.default_config)
    config_parser.set_defaults(default_config=config_args.default_config)
    
    parser = argparse.ArgumentParser(description="Hierarchical Argument Parser", parents=[config_parser])
    commands = parser.add_subparsers(title="Sub-commands")
    commands = parse_rag_command(commands, yaml_config)

    args = parse_args(parser, commands)
    pretty_print_args(args)

    main(args)

