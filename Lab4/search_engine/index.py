from typing import Dict, List, Tuple
from math import log

from .parser import Document

InvertedIndex = Dict[str, Dict[str, int]]


def build_inverted_index(docs: Dict[str, Document]) -> InvertedIndex:
    inverted: InvertedIndex = {}

    for doc_id, doc in docs.items():
        tf: Dict[str, int] = {}
        for w in doc.words:
            tf[w] = tf.get(w, 0) + 1

        for term, freq in tf.items():
            if term not in inverted:
                inverted[term] = {}
            inverted[term][doc_id] = freq

    return inverted


def compute_idf(inverted: InvertedIndex, num_docs: int) -> Dict[str, float]:
    idf: Dict[str, float] = {}
    for term, postings in inverted.items():
        df = len(postings)
        if df == 0:
            continue
        idf[term] = log(num_docs / df)
    return idf


def pretty_print_index(inverted: InvertedIndex) -> None:
    print("=== Инвертированный индекс ===")
    for term, postings in sorted(inverted.items()):
        print(f"{term!r}: {postings}")
