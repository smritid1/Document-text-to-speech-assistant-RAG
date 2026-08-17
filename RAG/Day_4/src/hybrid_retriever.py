# Hybrid search: combine BM25 (keyword) and vector (meaning) search results.
#
# The tricky part of combining two retrievers is that their scores aren't on
# the same scale -- a cosine distance of 0.3 and a BM25 score of 8.4 can't be
# averaged together meaningfully. Reciprocal Rank Fusion (RRF) sidesteps this
# by ignoring the raw scores entirely and combining based on RANK POSITION
# instead: "this chunk was #1 in list A and #3 in list B" is comparable
# information regardless of what scoring system produced each list.

from src.bm25_index import search_bm25
from src.settings import settings

RRF_K = 60  # standard constant from the original RRF paper -- dampens the
            # influence of any single list, rarely needs tuning


def _chunk_key(chunk: dict) -> tuple:
    meta = chunk["metadata"]
    return (meta.get("source"), meta.get("page"), chunk["text"][:50])


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], top_k: int) -> list[dict]:
    scores: dict[tuple, float] = {}
    chunk_lookup: dict[tuple, dict] = {}
    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            key = _chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            chunk_lookup[key] = chunk

    ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
    return [{**chunk_lookup[k], "rrf_score": scores[k]} for k in ranked_keys]


def hybrid_retrieve(vectorstore, query: str, top_k: int = None) -> list[dict]:

    top_k = top_k or settings.top_k
    candidate_k = top_k * 3 # pull a wider pool from each method before fusing
    vector_results = _vector_search(vectorstore, query, candidate_k)
    keyword_results = search_bm25(query, candidate_k)
    print(f"[vector] {len(vector_results)} results, [bm25] {len(keyword_results)} results")

    return reciprocal_rank_fusion([vector_results, keyword_results], top_k)


def _vector_search(vectorstore, query: str, top_k: int) -> list[dict]:
    """Same shape as retrieve_chunks() in rag.py."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return [{"text": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in results]