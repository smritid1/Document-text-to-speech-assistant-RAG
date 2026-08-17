
from pydantic import BaseModel, Field
 
 
class EvalItem(BaseModel):
    question: str
    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Ground-truth chunk_id(s) that answer this question. "
                     "Used for Recall@K / Precision@K / MRR.",
    )
    reference_answer: str | None = Field(
        default=None,
        description="A gold answer, if available. Needed for RAGAS context_recall "
                     "and is helpful (not required) for answer-quality checks.",
    )
 
 
class RetrievalResult(BaseModel):
    question: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
 
 
class RagasRow(BaseModel):
    """One row of the table RAGAS expects."""
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None
 