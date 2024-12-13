import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def irrelevant_removal(items, ndoc=None, key='full'):
    """ setting criteria to exclude the retrieved top-k 

    items: List of document item. keys include title, text, summary, ...
    """
    to_return = []
    for item in items[:ndoc]:
        ## criteria of inclusion 
        ### 1: text include irrelevant
        #### [BUG] some of the provided eval data have no summary field.....
        if key not in item.keys():
            item[key] = (item.get('extraction', None) or item.get('text'))
            logger.warn(f"NO `{key}`, use `extraction` or `text` instead") 

        # since ALCE additionally check the relevance when generating summary. 
        if "irrelevant" in item[key] or "Irrelevant" in item[key]:
            continue
            # item[key] = item['text']
            # to_return.append(item) 

        # ### 2: relevance score less than threshold
        # if ("relevance" in item) and (item["relevance"] < 0.0):
        #     continue
        to_return.append(item)

    logger.warn(f"Document removal: ({len(items)}) --> ({len(to_return)}).")
    return to_return
