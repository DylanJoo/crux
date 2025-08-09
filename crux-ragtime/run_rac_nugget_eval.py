import argparse

from evaluation.prejudge.retrieval_augmentation_context import rac_evaluate
from tools.neuclir.load_rac_data import load_rac_data


# Load data (corpus, initial query (report request), sub-questions), 
from tools.neuclir.ir_utils import (
    load_diversity_qrels,
    load_qrels, 
    load_query
)
from tools.neuclir.load_judgements import load_human_judgements, load_judgements

        
def run_rac_nugget_eval(args):

    queries, queries_for_search, raw_topics = load_query(args)

    # QRELS: 
    ## the document qrels are not available on GRID. download from TREC website:
    ## wget https://trec.nist.gov/data/neuclir/2023/neuclir-2023-qrels.final.tar.gz
    qrels = load_qrels(args.qrels)
    crux_qrels = load_qrels(args.crux_qrels)

    # Load RAC data. This includes running search or reading retrieved documents from a local file
    rac_data = load_rac_data(args, queries, queries_for_search, raw_topics,
                               retrieval_service_name=args.service_name)

    # We can either (1) covert the human labels to judgements, or (2) run CRUX pipeline to generate ratings.
    # judgements = load_judgements(args.data.judgement_file) if args.data.judgement_file is not None else None
    human_judgements = load_human_judgements(args.nuggets_dir)
    crux_judgements = load_judgements(args, rac_data)

    try:
        crux_judgements_andor = load_judgements(args, rac_data, load_andor=True)

        # Take MIN of regular and AND/OR judgements
        # Intuition: If context doesn't answer a question in general, it
        # shouldn't answer a question with an answer better

        for topicid in crux_judgements.keys():
            crux_judgements_topic = crux_judgements[topicid]
            crux_judgements_andor_topic = crux_judgements_andor[topicid]
            for docid in crux_judgements_topic.keys():
                crux_judgement = crux_judgements_topic[docid]
                crux_judgement_andor = crux_judgements_andor_topic[docid]
                if crux_judgement is None or crux_judgement_andor is None:
                    continue
                crux_judgement_andor_min = [min(a, b) for a, b in zip(crux_judgement, crux_judgement_andor)]
                crux_judgements_andor[topicid][docid] = crux_judgement_andor_min

    except:
        crux_judgements_andor = None

    output_eval = {}
    output_eval["human"] = rac_evaluate(
        args, qrels,
        human_judgements,
        rac_data=rac_data,
        diversity_qrels_path=args.qrels,
        tokenizer_name='bert-base-uncased',
        gamma=0.5, tag='experiment'
    )
    output_eval["crux_0620"] = rac_evaluate(
        args, crux_qrels,
        crux_judgements,
        rac_data=rac_data,
        diversity_qrels_path=args.crux_qrels,
        tokenizer_name='bert-base-uncased',
        gamma=0.5, tag='experiment'
    )

    if crux_judgements_andor is not None:
        output_eval["crux_andor_0620"] = rac_evaluate(
            args, crux_qrels,
            crux_judgements_andor,
            rac_data=rac_data,
            diversity_qrels_path=args.crux_qrels,
            tokenizer_name='bert-base-uncased',
            gamma=0.5, tag='experiment'
        )
    print("========== NeuCLIR 2024 (ReportGen) - Human Scores ==========")
    print(f"A-nDCG@20: {output_eval['human']['alpha_nDCG']}")
    print(f"Cov@10: {output_eval['human']['mean_coverage_at_10']}")
    print("\n========== NeuCLIR 2024 (ReportGen) - CRUX-0620 (100K qrels) Scores ==========")
    print(f"A-nDCG@20: {output_eval['crux_0620']['alpha_nDCG']}")
    print(f"Cov@10: {output_eval['crux_0620']['mean_coverage_at_10']}")
    if crux_judgements_andor is not None:
        print(f"AND/OR Cov@10: {output_eval['crux_andor_0620']['mean_coverage_at_10']}")
    #pprint.pprint(output_eval)

def main():

    parser = argparse.ArgumentParser()

    # ===========================================
    # ==== parameters for nugget evaluation =====
    # ===========================================
    parser.add_argument("--crux_dir", type=str, default="", help="Path where CRUX files are saved on disk")
    parser.add_argument("--crux_artifacts_path", type=str, default="/exp/scale25/artifacts/crux/crux-neuclir/crux_llama3.3-70b-instruct.jsonl", help="Path to CRUX artifacts jsonl file")
    # for additional AND/OR coverage metric use --crux_artifacts_path /exp/jturley/crux-scale/crux_andor_llama3.3-70b-instruct.jsonl
    parser.add_argument("--topics_path", type=str, default="/exp/scale25/neuclir/topics/neuclir24-test-request.jsonl", help="Path to topics file")
    parser.add_argument("--qrels", type=str, default="/exp/scale25/neuclir/eval/qrel/neuclir24-test-request.qrel", help="Path to qrel file")
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

    args, _ = parser.parse_known_args()
    
    run_rac_nugget_eval(args)
    
if __name__ == "__main__":
    main()
