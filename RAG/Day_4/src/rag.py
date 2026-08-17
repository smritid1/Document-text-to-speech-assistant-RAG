import re
import os

from langchain_community.llms import Ollama

from src.prompt import build_prompt
from src.schemas import ChatAnswer
from src.settings import settings
from src.hybrid_retriever import hybrid_retrieve
from src.history import retrieve_question

llm = Ollama(
    model=settings.llm_model,
    temperature=settings.temperature,
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
)


def ask_question(vectordb, question, history) -> ChatAnswer:
    history = history or []
    standalone_question = retrieve_question(question, history, llm)
    chunks = retrieve_chunk(vectordb, standalone_question)
    prompt = build_prompt(standalone_question, chunks)
    raw_answer = llm.invoke(prompt)

    # Pull "Pages: [...]" back out of the raw text -- this was missing
    # entirely, which is why it printed inline with the answer.
    pages = []
    match = re.search(r"Pages:\s*\[(.*?)\]", raw_answer)
    if match:
        # note: only pulls digits -- a page range like "53-54" from your
        # actual output above will NOT parse cleanly here, see note below
        pages = [int(p) for p in match.group(1).split(",") if p.strip().isdigit()]
        raw_answer = raw_answer[:match.start()].strip()

    return ChatAnswer(question=question, answer=raw_answer, pages=pages)


def retrieve_chunk(vectordb, question: str):
    if settings.use_hybrid_search:
        return hybrid_retrieve(vectordb, question, top_k=3)
    result = vectordb.similarity_search_with_score(question)
    chunks = []
    for doc, score in result:
        chunks.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
        })
    return chunks