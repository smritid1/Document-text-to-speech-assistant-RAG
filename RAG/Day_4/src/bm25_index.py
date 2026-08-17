
import pickle
import re
from pathlib import Path
 
from rank_bm25 import BM25Okapi
 
from src.settings import settings, PROJECT_ROOT
 
BM25_PATH = PROJECT_ROOT / settings.bm25_index_path
 
# Common words that appear in almost every chunk and carry no real meaning.
# Without removing these, a query and a chunk can "match" purely because
# they both contain "a" or "the" -- a false signal that pollutes fusion.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "of",
    "to", "for", "and", "or", "but", "with", "by", "as", "it", "this",
    "that", "what", "how", "does", "do", "did", "can", "be", "been",
}
 
 
def _tokenize(text: str) -> list[str]:

    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if w not in _STOPWORDS]
 
 
def build_bm25_index(texts: list[str], metadatas: list[dict]) -> None:
    tokenized_corpus = [_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
 
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts, "metadatas": metadatas}, f)
    print(f"BM25 index built: {len(texts)} chunks -> {BM25_PATH}")
 
 
def load_bm25_index() -> dict:
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"No BM25 index found at {BM25_PATH}. "
            f"Did you call build_bm25_index() during ingestion?"
        )
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)
 
 
def search_bm25(query: str, top_k: int = None) -> list[dict]:
  
    top_k = top_k or settings.top_k
    index = load_bm25_index()
    bm25, texts, metadatas = index["bm25"], index["texts"], index["metadatas"]
 
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []  # nothing meaningful to search on (e.g. query was all stopwords)
 
    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
 
    results = []
    for i in ranked_indices[:top_k]:
        if scores[i] <= 0:
            break  # scores are sorted descending, so everything after this is also 0
        results.append({"text": texts[i], "metadata": metadatas[i], "score": float(scores[i])})
    return results
 