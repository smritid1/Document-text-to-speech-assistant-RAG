
from src.evaluation.schemas import RetrievalResult
 
 
def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Of all relevant chunks, what fraction showed up in the top K retrieved?"""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & set(relevant_ids))
    return hits / len(relevant_ids)
 
 
def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Of the top K retrieved chunks, what fraction were actually relevant?"""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)
 
 
def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1 / (rank of first relevant chunk). 0 if none found."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0
 
 
def mean_reciprocal_rank(results: list[RetrievalResult]) -> float:
    """MRR across a whole eval set — how quickly, on average, does the
    first correct chunk show up in the ranked list?"""
    if not results:
        return 0.0
    scores = [reciprocal_rank(r.retrieved_chunk_ids, r.relevant_chunk_ids) for r in results]
    return sum(scores) / len(scores)
 
 
def evaluate_retrieval(results: list[RetrievalResult], k_values: list[int] = [1, 3, 5]) -> dict:
    """Aggregate Recall@K / Precision@K over multiple K values, plus MRR.
 
    Returns a dict like:
        {
          "recall@1": 0.42, "precision@1": 0.80,
          "recall@3": 0.71, "precision@3": 0.55,
          "recall@5": 0.85, "precision@5": 0.40,
          "mrr": 0.63,
          "n_queries": 20
        }
    """
    if not results:
        return {"n_queries": 0}
 
    metrics = {}
    for k in k_values:
        recalls = [recall_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k) for r in results]
        precisions = [precision_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k) for r in results]
        metrics[f"recall@{k}"] = sum(recalls) / len(recalls)
        metrics[f"precision@{k}"] = sum(precisions) / len(precisions)
 
    metrics["mrr"] = mean_reciprocal_rank(results)
    metrics["n_queries"] = len(results)
    return metrics
 