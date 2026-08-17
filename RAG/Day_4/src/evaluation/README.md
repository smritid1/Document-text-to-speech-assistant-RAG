Evaluating the RAG pipeline

A RAG system has two failure surfaces, and they need different metrics:

Retrieval — did the vector search pull back the right chunks?
Generation — given those chunks, did the LLM produce a good, grounded answer?

A system can fail at either stage independently. Good retrieval + bad generation looks like: relevant chunks were found, but the LLM ignored them or hallucinated anyway. Bad retrieval + good generation looks like: the LLM faithfully and fluently answers using the wrong chunks. Measuring only the final answer (which is tempting, since it's the only visible output) hides which stage is actually broken.

Retrieval metrics (src/evaluation/retrieval_metrics.py)

These need a ground-truth chunk_id per question — the chunk(s) that actually contain the answer. That's why chunking.py now stamps every chunk with a stable chunk_id ("{source}::p{page}::c{index}").

Metric	Question it answers
Recall@K	Of all the chunks that could answer this, what fraction did we find in the top K?
Precision@K	Of the top K chunks we retrieved, what fraction were actually relevant?
MRR	On average, how far down the ranked list is the first correct chunk? (1.0 = always first)

Recall and Precision trade off against each other via K: bigger K → recall goes up (more chances to include the right chunk) but precision goes down (more junk mixed in, which then gets stuffed into the LLM's context). MRR is useful because it's insensitive to K — it just asks "how good is our ranking," which is exactly what similarity_search is supposed to be good at.

Where the ground truth comes from: synthetic_dataset.py samples chunks directly from your indexed vectorstore and asks the local LLM to write one question per chunk that only that chunk answers. This is a fast way to get a non-trivial eval set with zero manual labeling, but it has a known bias: synthetic questions are generated from the chunk text, so they tend to reuse its vocabulary — which makes retrieval look easier than it will be for real users who phrase things differently. Treat the synthetic numbers as a ceiling, and replace/supplement with a handful of real questions (10-20) once you have any usage logs.

RAGAS metrics (src/evaluation/ragas_eval.py)

RAGAS scores the generation stage — it needs an LLM to act as a judge. This project wires the judge to the same local Ollama model already used for generation (settings.llm_model), so no external API key is required. That's a real trade-off: a small local model is a noisier, less consistent judge than GPT-4 or Claude would be. Use these scores to compare versions of your own pipeline (e.g. did switching chunk size from 500→1000 help?), not as absolute quality claims.

Metric	Question it answers	Needs ground truth?
Faithfulness	Does every claim in the answer trace back to something actually in the retrieved context? (catches hallucination)	No
Answer Relevancy	Does the answer actually address what was asked, rather than being generic or off-topic?	No
Context Precision	Of the chunks retrieved, were the relevant ones ranked near the top?	No (uses the generated answer as a proxy)
Context Recall	Did retrieval bring back everything needed to fully reconstruct the reference answer?	Yes — needs reference_answer

run_eval.py automatically skips Context Recall if any eval item is missing a reference_answer (dropping just that one metric rather than failing the whole run).

Running it
bash
# One-time: generate a synthetic eval set from whatever's currently indexed
python -m src.evaluation.run_eval --generate-dataset --n-samples 15

# Full run: retrieval metrics + RAGAS, against your real pipeline
python -m src.evaluation.run_eval --dataset eval_dataset.json

# Fast iteration loop: retrieval metrics only, no LLM judge calls
python -m src.evaluation.run_eval --dataset eval_dataset.json --skip-ragas

Install the added dependency:

bash
pip install ragas
Performance optimization

Once you have baseline numbers, here's what each low score typically points to, and the levers worth pulling:

Low Recall@K (right chunk isn't even in the candidate pool)

Chunk size too large → the answer is diluted inside a chunk full of other content, hurting the embedding match. Try smaller chunks (e.g. 500→300) with proportional overlap.
Chunk size too small → the answer gets split across chunk boundaries and no single chunk fully represents it. Try increasing overlap instead of shrinking size further.
Embedding model mismatch → generic sentence embeddings can miss domain-specific terminology (legal, medical, technical). Consider a domain-tuned or larger embedding model.
Consider hybrid search (BM25 keyword search + vector search, merged via reciprocal rank fusion) — catches exact-term matches (names, numbers, IDs) that dense embeddings often miss.

Low Precision@K but decent Recall@K (right chunk is found, just buried)

Add a re-ranker stage: retrieve top 20-30 with the fast vector search, then re-score with a cross-encoder re-ranker and keep the top 3-5. This is usually the single highest-ROI addition to a basic RAG pipeline.
Reduce K passed to the LLM once precision is fixed — less noise in context improves faithfulness too.

Low Faithfulness (hallucination despite good context)

Tighten the prompt: explicitly instruct "if it's not in the context, say you don't know" (you already do this in prompt.py — good). Consider lowering temperature further (already 0 by default via settings, verify).
Reduce the number of chunks stuffed into context — too much irrelevant surrounding text gives the LLM room to "connect dots" that aren't really connected.
Try a stronger generation model if available; faithfulness correlates with model capability more than most other metrics.

Low Answer Relevancy

Usually a prompt problem, not a retrieval problem — check whether the model is padding with generic preamble or answering a nearby-but-different question. Few-shot examples like the ones already in prompt.py help here.

Latency / throughput (not covered by RAGAS, but part of "performance")

Embed once, reuse: currently create_vectordb re-embeds on every "Load PDF" click even for the same file — consider hashing the file and skipping re-indexing if unchanged.
Batch embedding calls instead of one-by-one where possible (Chroma's from_texts already batches internally, but check embedding model batch size).
Persisted Chroma directory grows unbounded across uploads — add a cleanup or per-document collection strategy if this is used with many PDFs over time.
Ollama generation is usually the slowest step — consider a smaller/quantized model for iteration and a larger one only for final answers, or stream tokens to the UI (Streamlit supports st.write_stream) so perceived latency drops even if actual latency doesn't.
A note on interpreting scores together

Recall@K and Context Precision can disagree — e.g. good Recall@5 (right chunk is somewhere in the top 5) but low Context Precision (it's ranked 4th out of 5, with 3 irrelevant chunks ahead of it). That's a ranking problem, not a search problem, and points specifically at adding a re-ranker rather than changing chunk size or embedding model.