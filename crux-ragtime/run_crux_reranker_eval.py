import argparse

import pprint

from run_rac_nugget_eval import run_rac_nugget_eval


def main():

    parser = argparse.ArgumentParser()

    # ===========================================
    # ==== parameters for nugget evaluation =====
    # ===========================================
    parser.add_argument("--crux_dir", type=str, default="", help="Path where CRUX files are saved on disk")
    parser.add_argument("--crux_artifacts_path", type=str, default="/exp/scale25/artifacts/crux/crux-neuclir/crux_llama3.3-70b-instruct.jsonl", help="Path to CRUX artifacts jsonl file")
    parser.add_argument("--topics_path", type=str, default="/exp/scale25/neuclir/topics/example-request.jsonl", help="Path to topics file")
    parser.add_argument("--qrels", type=str, default="/exp/scale25/neuclir/eval/qrel/example-request.qrel", help="Path to qrel file")
    parser.add_argument("--crux_qrels", type=str, default="/exp/scale25/artifacts/crux/crux-neuclir/crux_llama3.3-70b-instruct.qrel", help="Path to crux qrel file")
    parser.add_argument("--nuggets_dir", type=str, default="/exp/scale25/neuclir/eval/nuggets", help="Path to qrel file")
    parser.add_argument("--model", type=str, default="llama3.3-70b-instruct", help="Model to use")
    parser.add_argument("--tag", type=str, default="human", help="Tag of run (for saving)") # 'human' if evaluating againt ref nuggets 
    parser.add_argument("--dataset_name", type=str, default="neuclir", help="Name of the dataset (for saving)")

    # search service parameters
    parser.add_argument("--host", default='https://scale25.hltcoe.org/', type=str)
    parser.add_argument("--port", default='443', type=str)
    parser.add_argument("--service_name", type=str, help="Retrieval service to use")
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--search_with_subqueries", default=False, action="store_true", help="When set to True, search will be performed using quer")

    # ===========================================
    # ==== parameters for rating generation =====
    # ===========================================
    
    parser.add_argument("--config", type=str, default="configs/scale.llama3-70b.yaml", help="Path to the config file")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=0)

    # Data
    parser.add_argument("--input", type=str, default=None, help="Path or directory of generated results")
    parser.add_argument("--split", type=str, default='test', help="Original split of datasets")
    parser.add_argument("--corpus_dir", type=str, default="/exp/scale25/neuclir/docs", help="Path to corpus")
    parser.add_argument("--output_dir", type=str, help="Path to generated results")

    # Model
    parser.add_argument("--load_mode", type=str, default='api', help="['vllm', 'api', 'litellm']")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=3, help="Max number of new tokens to generate in one step")
    
    # Other parameters
    parser.add_argument("--generate_additional_subtopics", default=False, action="store_true", help="Will generate subtopics in addition to the reference ones")
    parser.add_argument("--check_overlap", action="store_false", help="When set to True, scores will only be computed for qid's found in qrel file")
    parser.add_argument("--crux_reranking", action="store_false", help="When set to True, CRUX reranking will be applied")

    args, _ = parser.parse_known_args()
    
    if args.input is None:
        args.input = args.crux_dir
    if args.output_dir is None and args.tag in ["human"]:
        args.output_dir = args.crux_dir
    
    run_rac_nugget_eval(args)
    
if __name__ == "__main__":
    main()
