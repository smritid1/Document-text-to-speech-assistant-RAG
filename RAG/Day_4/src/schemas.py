# Pydantic models. Think of these as "labeled boxes" that describe exactly
# what fields our data should have and what type each field is.
# This helps catch mistakes early, e.g. putting text where a number should go.

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Info we store alongside every chunk of text, so we know where it came from."""
    source: str   # name of the PDF file
    page: int     # which page number this chunk came from


class ChatAnswer(BaseModel):
    """The final answer we give back to the user."""
    question: str
    answer: str
    pages: list[int]   # page numbers the answer was based on
