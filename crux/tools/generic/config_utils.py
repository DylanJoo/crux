import json
import sys
import yaml
import argparse

def get_random_test_qids():
    qids = [3936, 3967, 3453, 3487, 4368, 1643, 1685, 3524, 3536, 3560, 1817, 1885, 1888, 2930, 2941, 2960, 4411, 4426, 4454, 4473, 702, 708, 744, 673, 683, 694, 696, 593, 2302, 2394, 1523, 1543, 1590, 4722, 4778, 1942, 1975, 1977, 1990, 4834, 4208, 3675, 877, 2115, 1120, 1433, 1435, 1446, 1475, 4510, 4590, 2445, 2479, 2480, 408, 1082, 311, 322, 343, 374, 4921, 4962, 4998, 2206, 2247, 3810, 3858, 115, 140, 3741, 3757, 3792, 4198, 1335, 1351, 3267, 223, 241, 2703, 2738, 2745, 2767, 2649, 2654, 2672, 2028, 2029, 2051, 13, 38, 43, 86, 99, 902, 940, 941, 945, 952, 2847, 2511]
    return [f"multi_news-test-{i}" for i in qids]

def load_yaml_config(file_path):
    """Load YAML configuration file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def parse_args(parser, commands):
    """Parse hierarchical command-line arguments."""
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)

    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)

    # Parse base arguments
    parser.parse_args(split_argv[0], namespace=args)

    # Parse subcommands
    for argv in split_argv[1:]:
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)

    return args

def parse_rag_command(commands, yaml_config):
    """Create subparsers and set default values from YAML."""
    data_parser = commands.add_parser("data")
    data_config = yaml_config['data']
    data_parser.add_argument("--index_dir", type=str, default=data_config['index_dir'])
    data_parser.add_argument("--corpus_dir", type=str, default=data_config['corpus_dir'])
    data_parser.add_argument("--topic_file", type=str, default=data_config['topic_file'])
    data_parser.add_argument("--qrels_file", type=str, default=data_config['qrels_file'])
    data_parser.add_argument("--judgement_file", type=str, default=data_config['judgement_file'])
    # crux-specific
    data_parser.add_argument("--n_questions", type=int, default=data_config['n_questions'])
    data_parser.add_argument("--threshold", type=int, default=data_config['threshold'])

    rt_parser = commands.add_parser("retrieval")
    rt_config = yaml_config['retrieval']
    rt_parser.add_argument("--k", type=int, default=rt_config['k'])
    rt_parser.add_argument("--model_class", type=str, default=rt_config['model_class'])
    # leanred dense/sparse retrieval
    rt_parser.add_argument("--model_name_or_path", type=str, default=rt_config['model_name_or_path'])
    rt_parser.add_argument("--max_length", type=int, default=rt_config['max_length'])
    rt_parser.add_argument("--pooling", type=str, default=rt_config['pooling'])
    rt_parser.add_argument("--l2_norm", default=rt_config['l2_norm'], action='store_true')
    rt_parser.add_argument("--batch_size", type=int, default=rt_config['batch_size'])
    # bm25 
    rt_parser.add_argument("--k1", type=float, default=rt_config['k1'])
    rt_parser.add_argument("--b", type=float, default=rt_config['b'])
    rt_parser.add_argument("--seed", type=int, default=0)

    rr_parser = commands.add_parser("reranking")
    rr_config = yaml_config['reranking']
    rr_parser.add_argument("--model_class", type=str, default=rr_config['model_class'])
    rr_parser.add_argument("--model_name_or_path", type=str, default=rr_config['model_name_or_path'])
    rr_parser.add_argument("--top_k", type=int, default=rr_config['top_k'])
    rr_parser.add_argument("--batch_size", type=int, default=rr_config['batch_size'])
    rr_parser.add_argument("--max_length", type=int, default=rr_config['max_length'])

    lw_parser = commands.add_parser("listwise_reranking")
    lw_config = yaml_config['listwise_reranking']
    lw_parser.add_argument("--model_name_or_path", type=str, default=lw_config['model_name_or_path'])
    lw_parser.add_argument("--max_k", type=int, default=lw_config['max_k'])
    lw_parser.add_argument("--batch_size", type=int, default=lw_config['batch_size'])
    lw_parser.add_argument("--max_length", type=int, default=lw_config['max_length'])
    lw_parser.add_argument("--use_logits", default=lw_config['use_logits'], action='store_true')
    lw_parser.add_argument("--use_alpha", default=lw_config['use_alpha'], action='store_true')
    lw_parser.add_argument("--num_passes", type=int, default=lw_config['num_passes'])
    lw_parser.add_argument("--system_message", type=str, default=lw_config['system_message'])
    lw_parser.add_argument("--type", type=str, default=lw_config['type'], choices=['listwise', 'setwise', 'mmr'])

    aug_parser = commands.add_parser("augmentation")
    aug_config = yaml_config['augmentation']
    aug_parser.add_argument("--type", type=str, default=aug_config['type'])
    aug_parser.add_argument("--max_k", type=int, default=aug_config['max_k'])
    aug_parser.add_argument("--batch_size", type=int, default=aug_config['batch_size'])
    aug_parser.add_argument("--max_length", type=int, default=aug_config['max_length'])

    gen_parser = commands.add_parser("generation")
    gen_config = yaml_config['generation']
    gen_parser.add_argument("--model_name_or_path", type=str, default=gen_config['model_name_or_path'])
    gen_parser.add_argument("--batch_size", type=int, default=gen_config['batch_size'])
    gen_parser.add_argument("--max_length", type=int, default=gen_config['max_length'])
    gen_parser.add_argument("--temperature", type=float, default=gen_config['temperature'])
    # gen_parser.add_argument("--think_activated", type=bool, action='store_true', default=gen_config['think_activated'])
    return commands

def pretty_print_args(args):
    """Neatly print out parsed arguments."""
    def recursive_namespace_to_dict(namespace):
        """Recursively convert argparse.Namespace to a dictionary."""
        if isinstance(namespace, argparse.Namespace):
            return {key: recursive_namespace_to_dict(value) for key, value in vars(namespace).items()}
        return namespace

    args_dict = recursive_namespace_to_dict(args)
    print(json.dumps(args_dict, indent=4))  # Pretty-print JSON format

if __name__ == "__main__":
    ## Here is the sample calling pipeline
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--default_config", type=str, default="configs/dummy.yaml", help="Path to the configuration YAML file")
    config_args, remaining_argv = config_parser.parse_known_args()
    
    # Step 3: Load the YAML config
    yaml_config = load_yaml_config(config_args.default_config)
    
    # Step 4: Create the main parser
    parser = argparse.ArgumentParser(description="Hierarchical Argument Parser", parents=[config_parser])
    commands = parser.add_subparsers(title="Sub-commands")
    commands = parse_rag_command(commands, yaml_config)

    # Step 5: Parse final arguments
    args = parse_args(parser, commands)
    retty_print_args(args)
