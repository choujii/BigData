from typing import Dict, List, Tuple
from math import log

from .index import InvertedIndex
from .parser import Document


def tokenize_query(q: str) -> List[str]:
    import re
    return [w.lower() for w in re.findall(r"\w+", q, flags=re.UNICODE)]


def taat_search(
        query: str,
        inverted: InvertedIndex,
        idf: Dict[str, float],
        num_docs: int
) -> List[Tuple[str, float]]:
    """
    Term-at-a-time:
    обходим термы запроса один за другим и аккумулируем score документов.
    """
    terms = tokenize_query(query)
    scores: Dict[str, float] = {}

    for term in terms:
        postings = inverted.get(term, {})
        df = len(postings)
        if df == 0:
            continue
        term_idf = idf.get(term, log(num_docs / df))

        for doc_id, tf in postings.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + tf * term_idf

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def daat_search(
        query: str,
        inverted: InvertedIndex,
        idf: Dict[str, float],
        num_docs: int
) -> List[Tuple[str, float]]:
    """
    Document-at-a-time:
    идём по документам, объединяя отсортированные postings-списки термов.
    """
    terms = tokenize_query(query)

    term_postings = []
    for term in terms:
        postings_dict = inverted.get(term, {})
        if not postings_dict:
            continue
        postings = sorted(postings_dict.items(), key=lambda x: x[0])
        term_postings.append((term, postings))

    if not term_postings:
        return []

    indices = [0] * len(term_postings)
    scores: Dict[str, float] = {}

    while True:
        current_docs = []
        for i, (_, postings) in enumerate(term_postings):
            if indices[i] < len(postings):
                current_docs.append(postings[indices[i]][0])

        if not current_docs:
            break

        min_doc = min(current_docs)

        score = 0.0
        for i, (term, postings) in enumerate(term_postings):
            while indices[i] < len(postings) and postings[indices[i]][0] < min_doc:
                indices[i] += 1

            if indices[i] < len(postings) and postings[indices[i]][0] == min_doc:
                tf = postings[indices[i]][1]
                df = len(postings)
                term_idf = idf.get(term, log(num_docs / df))
                score += tf * term_idf
                indices[i] += 1

        scores[min_doc] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def apply_pagerank_boost(
        ranked: List[Tuple[str, float]],
        pagerank: Dict[str, float],
        alpha: float = 0.8
) -> List[Tuple[str, float]]:
    """
    Комбинированный скор: final = alpha * text_score + (1 - alpha) * norm_pagerank
    """
    if not ranked:
        return ranked

    max_pr = max(pagerank.values()) if pagerank else 1.0
    boosted: List[Tuple[str, float]] = []

    for doc_id, score in ranked:
        pr = pagerank.get(doc_id, 0.0) / max_pr if max_pr > 0 else 0.0
        final_score = alpha * score + (1 - alpha) * pr
        boosted.append((doc_id, final_score))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted
