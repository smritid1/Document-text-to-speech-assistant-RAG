from pydantic import BaseModel
from enum import Enum
class ChunkMetadata(BaseModel):
    source: str 
    page: int
    chunk_id: str = ""

class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class ChatMessage(BaseModel):
    role: ChatRole
    content: str