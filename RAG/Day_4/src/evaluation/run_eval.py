
import json
 
from src.vector_database import load_vectordb
from src.rag import retrieve_chunk, build_prompt
from src.rag import llm as generation_llm
from src.evaluation.schemas import EvalItem, RetrievalResult, RagasRow
from src.evaluation.retrival_metrics import evaluate_retrieval
from src.evaluation.ragas_eval import run_ragas_eval
from src.evaluation.synthetic_dataset import (
    generate_synthetic_dataset,
    save_dataset,
    load_dataset,
)
 
 
 
DATASET_PATH = "/home/ishan/Desktop/RAG/eval_dataset.json"          # where the eval set lives / will be saved
GENERATE_DATASET = True                     # True -> build a synthetic eval set first
N_SAMPLES = 15                                # how many synthetic questions to generate
SKIP_RAGAS = False                            # True -> only run retrieval metrics (fast)
REPORT_OUT = "eval_report.json"              # where the final report is saved
K_VALUES = [1, 3, 5]                          # which K values to compute Recall@K/Precision@K for
 
 
 
def run_pipeline_on_dataset(vectordb, dataset: list[EvalItem]):
    retrieval_results = []
    ragas_rows = []
 
    for item in dataset:
        chunks = retrieve_chunk(vectordb, item.question) 
        retrieved_ids = [c["metadata"].get("chunk_id", "") for c in chunks]
 
        prompt = build_prompt(item.question, chunks)
        answer = generation_llm.invoke(prompt)
 
        retrieval_results.append(
            RetrievalResult(
                question=item.question,
                retrieved_chunk_ids=retrieved_ids,
                relevant_chunk_ids=item.relevant_chunk_ids,
            )
        )
        ragas_rows.append(
            RagasRow(
                question=item.question,
                answer=answer,
                contexts=[c["text"] for c in chunks],
                ground_truth=item.reference_answer,
            )
        )
    return retrieval_results, ragas_rows
 
 
def main():
    vectordb = load_vectordb()
    if GENERATE_DATASET:
        print(f"Generating synthetic eval set ({N_SAMPLES} samples)...")
        dataset = generate_synthetic_dataset(vectordb, n_samples=N_SAMPLES)
        save_dataset(dataset, DATASET_PATH)
        print(f"Saved {len(dataset)} items to {DATASET_PATH}")
    else:
        dataset = load_dataset(DATASET_PATH)
 
    print(f"Running pipeline on {len(dataset)} questions...")
    retrieval_results, ragas_rows = run_pipeline_on_dataset(vectordb, dataset)
 
    print("\n=== Retrieval metrics ===")
    retrieval_report = evaluate_retrieval(retrieval_results, k_values=K_VALUES)
    for k, v in retrieval_report.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
 
    report = {"retrieval": retrieval_report}
    if not SKIP_RAGAS:
        print("\n=== RAGAS metrics (this calls the local LLM once per question per metric — slow) ===")
        has_ground_truth = all(r.ground_truth for r in ragas_rows)
        ragas_result = run_ragas_eval(ragas_rows, include_context_recall=has_ground_truth)
        print(ragas_result)
        report["ragas"] = str(ragas_result)
 
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {REPORT_OUT}")
 
 
if __name__ == "__main__":
    main()
 