from datetime import datetime

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    id: str
    filename: str
    page_count: int
    processing_status: str
    extraction_method: str
    created_at: datetime
    file_size: int


class ChatQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class CompareQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[str] = Field(..., min_length=2, max_length=5)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class DocumentPage(BaseModel):
    document_id: str
    page_number: int
    text: str
    extraction_method: str
    ocr_confidence: float | None = None
    extraction_error: str | None = None
    processing_status: str


class ChatAnswer(BaseModel):
    answer: str
    citations: list[dict]
    passages: list[dict]
    source_documents: list[str]
    generation_mode: str = "extractive"
    provider: str | None = None
    model: str | None = None
    insufficient_evidence: bool = False
    provider_error: str | None = None


class VisualQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_status: dict
