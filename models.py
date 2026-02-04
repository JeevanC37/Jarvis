"""Pydantic models for API requests and responses."""
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's message/query")
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous conversation messages for context"
    )
    use_knowledge_base: bool = Field(
        default=True,
        description="Whether to retrieve context from knowledge base"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Assistant's response")
    sources: Optional[List[dict]] = Field(
        default=None,
        description="Source documents used for the response"
    )


class KnowledgeAddRequest(BaseModel):
    """Request model for adding knowledge."""
    doc_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content to store")
    metadata: Optional[dict] = Field(
        default=None,
        description="Additional metadata for the document"
    )


class KnowledgeSearchRequest(BaseModel):
    """Request model for searching knowledge."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, description="Number of results to return")


class KnowledgeResponse(BaseModel):
    """Response model for knowledge operations."""
    status: str
    message: str
    data: Optional[dict] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    llm_status: dict
    vector_db_status: str
