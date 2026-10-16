from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SourceDocument(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt or query")
    session_id: Optional[str] = Field(default=None, description="Unique conversation session ID")
    stream: bool = Field(default=False, description="Whether to stream response tokens")
    use_rag: bool = Field(default=True, description="Whether to use RAG knowledge retrieval")
    provider: Optional[str] = Field(default=None, description="LLM provider: openai, gemini, or local")
    system_prompt: Optional[str] = Field(default=None, description="Optional override for system prompt")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    session_id: str
    message: str
    sources: List[SourceDocument] = Field(default_factory=list)
    provider: str
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
