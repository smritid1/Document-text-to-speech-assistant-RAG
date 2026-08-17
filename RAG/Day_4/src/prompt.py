from src.schema import ChatMessage


def _format_history(history: list[ChatMessage]) -> str:
    """Turn the last few turns into readable text for the prompt.
    Kept as its own function so both the prompt and (later, if you want)
    the condensation step can format history the exact same way."""
    if not history:
        return "(no prior conversation)"
    recent = history[-6:]  # last 3 user/assistant pairs -- keeps prompt size bounded
    return "\n".join(f"{m.role.value.capitalize()}: {m.content}" for m in recent)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Combine retrieved chunks + question + conversation history into one
    prompt for the LLM, using a few examples (few-shot learning) to show
    the model exactly the style and level of detail we want in its answers."""


    context = ""
    for chunk in chunks:
        page = chunk["metadata"]["page"]
        context += f"[Page {page}] {chunk['text']}\n\n"

    examples = """Example 1:
Context:
[Page 3] The water cycle describes how water moves through Earth's systems. It begins with evaporation, where the sun heats water in oceans and lakes, turning it into vapor. This vapor rises and cools, forming clouds through condensation.
[Page 4] Once clouds become heavy with condensed water, precipitation occurs in the form of rain, snow, or hail. This water then flows back into rivers and oceans through runoff, completing the cycle.

Question: How does the water cycle work?
Answer: The water cycle works through a continuous four-stage process. First, the sun's heat causes evaporation, turning water from oceans and lakes into vapor. Second, this vapor rises into the atmosphere and cools down, undergoing condensation to form clouds. Third, once the clouds accumulate enough water, precipitation releases it back to the earth as rain, snow, or hail. Finally, this water flows through runoff into rivers and oceans, where the entire cycle begins again.
Pages: [3, 4]

Example 2:
Context:
[Page 7] Photosynthesis is the process by which plants convert light energy into chemical energy. Chlorophyll in the leaves absorbs sunlight, while the plant takes in carbon dioxide through small pores called stomata and water through its roots.
[Page 8] Using the absorbed sunlight, the plant combines carbon dioxide and water to produce glucose, which it uses for energy and growth. Oxygen is released as a by-product of this reaction.

Question: What does a plant need for photosynthesis?
Answer: A plant needs three main things for photosynthesis: sunlight, carbon dioxide, and water. Chlorophyll in the leaves captures the sunlight, while carbon dioxide enters through tiny pores called stomata and water is absorbed through the roots. The plant then combines these ingredients to produce glucose for energy and growth, releasing oxygen as a by-product in the process.
Pages: [7, 8]

Example 3:
Context:
[Page 2] Quarterly revenue figures were not disclosed in this section of the report.

Question: What was the exact revenue for Q3?
Answer: I don't know based on the provided document. The context available does not include specific revenue figures for Q3.
Pages: []
"""

    prompt = f"""You are a careful assistant that answers questions using only the given context.
Study the examples below to see the level of detail expected: explain your reasoning,
connect the relevant pieces of context together, and write a few full sentences rather
than a single short phrase. Do not use any outside knowledge.
If the answer is not in the context, say "I don't know" and briefly explain why.
At the end of every answer, write the page numbers you used, like this: Pages: [1, 2]
If you could not answer, write: Pages: []



{examples}
Conversation so far:


Now answer this new question the same way, using the conversation above only to
understand what the question is referring to -- your answer must still come only
from the Context below, not from anything said earlier in the conversation.

Context:
{context}
Question: {question}
Answer:"""

    return prompt