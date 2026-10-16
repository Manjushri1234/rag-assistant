"""Request and response models for the API."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class Source(BaseModel):
    id: int
    doc_name: str
    section: str
    source_url: str
    score: float


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    status: Literal["answered", "no_answer"]
    sources: list[Source]
    query_type: str
    retries: int
    queries_tried: list[str]
    grounding: Literal["supported", "unsupported", "unverified", "not_applicable"]
    grounding_note: str
    generation_attempts: int


class IngestUrlRequest(BaseModel):
    url: HttpUrl
    name: Optional[str] = Field(None, max_length=60)


class IngestResponse(BaseModel):
    doc_name: str
    chunks: int


class DocumentInfo(BaseModel):
    doc_name: str
    source_url: str
    chunks: int


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(None, max_length=1000)


class FeedbackResponse(BaseModel):
    status: str = "recorded"