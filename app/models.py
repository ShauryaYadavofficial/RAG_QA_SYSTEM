import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


# ── Upload ─────────────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    message: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    chunk_count: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ── Query ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The question to answer",
    )
    document_ids: Optional[list[str]] = Field(
        default=None,
        description="Limit retrieval to specific document IDs. "
                    "Omit to search all documents.",
    )
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be blank.")
        return v.strip()


class RetrievedChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    similarity_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    latency_ms: float
    model_used: str


# ── Internal state store ───────────────────────────────────────────────────

class DocumentRecord(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_path: str
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)