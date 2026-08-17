
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
 
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
 
from Day_4.src.settings import settings
from src.evaluation.schemas import RagasRow
 
 
def _build_judge():
    llm = Ollama(
        model=settings.llm_model,
        temperature=0,  # deterministic scoring
        base_url=settings.get("ollama_base_url", "http://localhost:11434"),
    )
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)
 
 
def run_ragas_eval(rows: list[RagasRow], include_context_recall: bool = True):
    """rows: one per question, with the answer your RAG pipeline actually
    produced and the contexts your retriever actually returned (not the
    ground truth chunks — the REAL retrieved ones, so faithfulness/relevancy
    reflect what really happened).
 
    context_recall requires a ground_truth/reference_answer on every row;
    if some rows lack one, set include_context_recall=False or those rows
    will fail scoring.
    """
    judge_llm, judge_embeddings = _build_judge()
 
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
    ]
    if include_context_recall:
        metrics.append(ContextRecall(llm=judge_llm))
 
    dataset = EvaluationDataset.from_list([
        {
            "user_input": r.question,
            "response": r.answer,
            "retrieved_contexts": r.contexts,
            "reference": r.ground_truth or "",
        }
        for r in rows
    ])
 
    result = evaluate(dataset=dataset, metrics=metrics)
    return result  
 