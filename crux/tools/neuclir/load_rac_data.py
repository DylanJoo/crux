import asyncio
from argparse import Namespace

from ..neuclir.create_run_file import create_run_file
from ..search.search import async_get_content, get_content, retrieve_with_report_request, retrieve_with_subqueries


def load_rac_data(args: Namespace, queries: dict, queries_for_search: dict, raw_topics: dict,
                  k: int = None, collection: str = "neuclir", 
                  retrieval_service_name: str = "plaidx-neuclir") -> dict:
    """Load RAC data
    Args:
        args [Namespace]: args provided as input
        queries [dict]: full queries (query + background)
        queries_for_search [dict]: queries to be used when querying search service
        k [int]: top-k documents to keep
        collection [str]: collection to be used when retrieving document
            content from search service
        retrieval_service_name [str]: retrieval service name
    """
    if not k:
        k = args.top_k
    rac_data = asyncio.run(_async_load_rac_data(args, queries, queries_for_search, raw_topics,
                                                k, collection, retrieval_service_name))
    return rac_data


async def _async_load_rac_data(args: Namespace, queries: dict, queries_for_search: dict, raw_topics: dict,
                               k: int = None, collection: str = "neuclir", 
                               retrieval_service_name: str = "plaidx-neuclir") -> dict:
    rac_data = {}

    qids, qs = zip(*queries.items())
    if args.search_with_subqueries:
        retrieved = await asyncio.gather(*[retrieve_with_subqueries(args, queries_for_search[qid], retrieval_service_name, top_k_docs=k) for qid in qids])
    else:
        retrieved = await asyncio.gather(*[retrieve_with_report_request(
            args, queries_for_search[qid], retrieval_service_name, 
            [t for t in raw_topics if t["request_id"]==qid][0], # get topic
            top_k_docs=k) for qid in qids])

    for qid, q, retrieved in zip(qids, qs, retrieved):
        # Get content for each retrieved document
        raw_content = await asyncio.gather(*[async_get_content(collection, doc_id) for doc_id in retrieved["result"][:k]])

        rac_data[qid] = {  # TODO: might be missing some fields
            "qid": qid,
            "topic": q,
            "questions": None,  # list_questions,
            "type": None,  # "vanilla", k) if max_k > 0 else ("oracle", k),
            "docids": list(retrieved["result"])[:k],  # retrieved top k
            "context_list": raw_content,
            "prompt": None,  # template_fn_mapping[template_type](documents),
            "report": None,
            "response": None,
        }

    if args.crux_dir:
        create_run_file(args, queries, rac_data)

    return rac_data
