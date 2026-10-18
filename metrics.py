"""Retrieval metrics. A result is a chunk dict: {"paper_id", "page", "text"}.
A question's gold is a list of {"paper_id", "page"}. Most questions have one gold place;
multi-hop questions have two or more."""


def is_hit(chunk, gold, page_slack=1):
    """Right paper and the right page (or a page next to it)."""
    return any(chunk["paper_id"] == g["paper_id"] and abs(chunk["page"] - g["page"]) <= page_slack
               for g in gold)


def first_hit_rank(results, gold, page_slack=1):
    """Position (starting at 1) of the first correct chunk, or None."""
    for rank, chunk in enumerate(results, 1):
        if is_hit(chunk, gold, page_slack):
            return rank
    return None


def recall_at_k(ranks, k):
    """ranks: one first-hit rank (or None) per question. Share of questions hit within top k."""
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mrr(ranks):
    """Mean of 1/rank. A miss counts as 0."""
    return sum(1 / r if r else 0 for r in ranks) / len(ranks)


def all_gold_in_top_k(results, gold, k, page_slack=1):
    """For multi-hop: did the top k contain EVERY gold place?"""
    top = results[:k]
    return all(any(is_hit(c, [g], page_slack) for c in top) for g in gold)


if __name__ == "__main__":
    A = {"paper_id": "a", "page": 3, "text": ""}
    B = {"paper_id": "b", "page": 1, "text": ""}
    gold = [{"paper_id": "a", "page": 3}]

    assert first_hit_rank([A, B], gold) == 1
    assert first_hit_rank([B, B, A], gold) == 3
    assert first_hit_rank([B, B], gold) is None
    assert first_hit_rank([{"paper_id": "a", "page": 4, "text": ""}], gold) == 1   # next page counts
    assert first_hit_rank([{"paper_id": "a", "page": 5, "text": ""}], gold) is None  # too far

    ranks = [1, 3, None]
    assert abs(recall_at_k(ranks, 1) - 1 / 3) < 1e-9
    assert abs(recall_at_k(ranks, 3) - 2 / 3) < 1e-9
    assert abs(mrr(ranks) - (1 + 1 / 3 + 0) / 3) < 1e-9

    two = [{"paper_id": "a", "page": 3}, {"paper_id": "b", "page": 1}]
    assert all_gold_in_top_k([A, B], two, 2)
    assert not all_gold_in_top_k([A, B], two, 1)
    print("all metric tests passed")
