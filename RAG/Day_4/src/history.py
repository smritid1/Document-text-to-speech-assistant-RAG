# Conversational memory, part 1: query condensation.
#
# The problem: if the user's first question is "What is RAG?" and the
# second is "How does that reduce hallucination?", embedding THAT SECOND
# QUESTION ALONE and searching with it fails -- "that" has no meaning to
# a vector search. The fix is to rewrite the follow-up into a standalone
# question BEFORE retrieval ever runs, using the chat history for context.

from src.schema import ChatMessage




_EXAMPLES = """
 Example 1 --> simple pronoun follow-up
 Chat History:
 User: What is a Thekuwa?
 Assistant: Thekuwa is a religious food that is prepared on the occasion of Chhath, which is mainly celebrated in the month of Kartik (October-November). It is described as a specific type of food that is prepared for this occasion. 
 Follow Up input: When is it eaten?
 Standalone_question: When is Thekuwa eaten?  

Example 2 -- topic switch, pronoun must attach to the MOST RECENT topic only:
Chat History:
User: What is Nepal?
Assistant: Nepal is a country in South Asia, home to Mount Everest.
User: What is cheese?
Assistant: Cheese is a dairy product made from milk.
Follow Up Input: Who eats it?
Standalone question: Who eats cheese?  

Example 3 -- vague reference to "that" pointing at the immediately prior answer, not an earlier one:
Chat History:
User: What is BM25?
Assistant: BM25 is a keyword-based ranking algorithm.
User: What is vector search?
Assistant: Vector search finds text by meaning using embeddings.
Follow Up Input: When should I use that instead of keyword matching?
Standalone question: When should I use vector search instead of keyword matching?

Example 4 -- already standalone (must NOT be changed):
Chat History:
User: What is RAG?
Assistant: RAG stands for Retrieval-Augmented Generation.
Follow Up Input: What is the capital of France?
Standalone question: What is the capital of France?

"""

def retrieve_question(question: str, history: list[ChatMessage], llm) -> str:
    """Rewrite a possibly-ambiguous follow-up question into a standalone
    question that includes whatever context it was implicitly relying on.
    If there's no history yet, there's nothing to condense -- return as-is.
    """
    if not history:
        return question

    recent_history = history[-6:]  # last 3 user/assistant pairs
    history_text = "\n".join(f"{m.role.value.capitalize()}: {m.content}" for m in recent_history)

    prompt = f"""

Given a conversation history and a follow-up question, rewrite the
follow-up question as a standalone question that includes all necessary context.

Rules:
- If the follow-up question is already standalone (doesn't depend on anything
  said before), return it completely unchanged.
- If the conversation has moved between multiple topics, resolve any pronoun
  ("it", "that", "this", "them") using ONLY the most recently discussed topic,
  never an earlier one -- even if the earlier topic is still mentioned above it.
- Do not add any information that wasn't asked for. Only resolve references,
  don't answer the question.
- Reply with **ONLY** the rewritten question and nothing else -- no explanation,
  no "Standalone question:" prefix in your reply.

{_EXAMPLES}

Conversation history:
{history_text}

Follow-up question: {question}

Standalone question:"""

    result = llm.invoke(prompt).strip()
    return result if result else question