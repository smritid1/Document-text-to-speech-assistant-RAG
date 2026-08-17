
import json
import random
 
from langchain_community.llms import Ollama
 
from Day_4.src.settings import settings
from src.evaluation.schemas import EvalItem
 
QUESTION_GEN_PROMPT = """You will be shown one excerpt from a document.
Write exactly ONE question that this excerpt answers directly and completely.
The question must be answerable using ONLY this excerpt — do not reference
other pages or outside knowledge. Also give a one-to-two sentence answer,
using only this excerpt.
 
Respond in EXACTLY this format, nothing else:
Question: <question>
Answer: <answer>
 
Excerpt:
\"\"\"
{chunk_text}
\"\"\"
"""
 
 
def _parse_question_answer(raw: str) -> tuple[str, str] | None:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    question, answer = None, None
    for line in lines:
        if line.lower().startswith("question:"):
            question = line.split(":", 1)[1].strip()
        elif line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
    if question and answer:
        return question, answer
    return None
 
 
def generate_synthetic_dataset(
    vectorstore,
    n_samples: int = 15,
    min_chunk_chars: int = 200,
    seed: int = 42,
) -> list[EvalItem]:
    """Sample chunks from the vectorstore's Chroma collection and generate
    one question per sampled chunk."""
 
    llm = Ollama(
        model=settings.llm_model,
        temperature=0.3,  # a bit of variety in question phrasing, but still grounded
        base_url=settings.get("ollama_base_url", "http://localhost:11434"),
    )
 
    raw_collection = vectorstore.get(include=["documents", "metadatas"])
    docs = raw_collection["documents"]
    metas = raw_collection["metadatas"]
    # filter out tiny/near-empty chunks — not enough signal for a good question
    candidates = [
        (doc, meta) for doc, meta in zip(docs, metas)
        if doc and len(doc) >= min_chunk_chars and meta.get("chunk_id")
    ]
 
    if not candidates:
        raise ValueError(
            "No chunks with chunk_id found in the vectorstore. "
            "Re-index your PDF with the updated chunking.py that assigns chunk_id."
        )
 
    random.seed(seed)
    sample = random.sample(candidates, k=min(n_samples, len(candidates)))
 
    dataset: list[EvalItem] = []
    for doc, meta in sample:
        prompt = QUESTION_GEN_PROMPT.format(chunk_text=doc)
        raw = llm.invoke(prompt)
        parsed = _parse_question_answer(raw)
        if parsed is None:
            continue
        question, answer = parsed
        dataset.append(
            EvalItem(
                question=question,
                relevant_chunk_ids=[meta["chunk_id"]],
                reference_answer=answer,
            )
        )
 
    return dataset
 
 
def save_dataset(dataset: list[EvalItem], path: str = "eval_dataset.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([item.model_dump() for item in dataset], f, indent=2, ensure_ascii=False)
 
 
def load_dataset(path: str = "eval_dataset.json") -> list[EvalItem]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [EvalItem(**item) for item in raw]