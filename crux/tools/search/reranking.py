import numpy as np

from augmentation.gen_ratings import gen_ratings, async_gen_ratings

async def rerank_with_crux(args, docs, topic, k):

    ratings = await async_gen_ratings(args, docs_to_rerank=docs, topic=topic)

    #ranked_docs = rank_by_sum_of_crux_scores(ratings)
    ranked_docs = rank_by_maximizing_coverage(ratings)

    return ranked_docs[:k]

def rank_by_sum_of_crux_scores(ratings):
    sums = [sum(row) for row in ratings["ratings"]]
    indexed_sums = list(enumerate(sums))
    ranked = sorted(indexed_sums, key=lambda x: x[1], reverse=True)
    ranked_indices = [index for index, _ in ranked]
    ranked_docs = [ratings["docids"][i] for i in ranked_indices]
    return ranked_docs

def rank_by_maximizing_coverage(ratings):
    
    rows = np.array(ratings["ratings"])

    # Compute max for each column
    col_max = np.max(rows, axis=0)

    # Track remaining rows and uncovered columns
    selected_order = []
    covered = np.zeros_like(col_max, dtype=bool)
    remaining_indices = set(range(len(rows)))

    while remaining_indices:
        # Find the row that covers the most new max values
        best_row = None
        best_gain = -1

        for i in remaining_indices:
            gain = np.logical_and(rows[i] == col_max, ~covered).sum()
            if gain > best_gain:
                best_gain = gain
                best_row = i

        selected_order.append(best_row)
        covered = np.logical_or(covered, rows[best_row] == col_max)
        remaining_indices.remove(best_row)

        if np.all(covered):
            break # no more gain can be added

    # When there is no more gain to be added, continue with
    # 'rank_by_sum_of_crux_scores' strategy
    while remaining_indices:
        best_row = None
        best_sum_scores = -1

        for i in remaining_indices:
            sum_scores = np.sum(rows[i])
            if sum_scores > best_sum_scores:
                best_sum_scores = sum_scores
                best_row = i

        selected_order.append(best_row)
        remaining_indices.remove(best_row)

    ordered_rows = [ratings["docids"][i] for i in selected_order]
    return ordered_rows
