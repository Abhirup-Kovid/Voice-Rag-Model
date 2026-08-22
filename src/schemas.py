"""Schemas for I/O and internal data structures."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal

class VoiceRequest(BaseModel):
    # Actually for voice it will be a multipart request, but we can define a schema if needed
    pass

class VoiceResponse(BaseModel):
    pass

class TextRequest(BaseModel):
    query: str
    language: Optional[str] = None

class Source(BaseModel):
    chunk_id: str
    text: str
    language: str
    strategy: str
    score: float
    doc_id: str
    char_offsets: Optional[List[int]] = None

class LatencyBreakdown(BaseModel):
    stt_ms: Optional[int] = None
    embed_ms: Optional[int] = None
    retrieve_ms: Optional[int] = None
    rerank_ms: Optional[int] = None
    guardrail_ms: Optional[int] = None
    llm_first_token_ms: Optional[int] = None
    llm_total_ms: Optional[int] = None
    total_ms: Optional[int] = None

class EvidenceLink(BaseModel):
    sentence: str
    source_idx: int
    source_id: str
    confidence: float

class GuardrailResult(BaseModel):
    passed: bool
    off_topic: bool
    unsafe: bool
    grounded: bool
    refused: bool
    reason: Optional[str] = None
    evidence_score: Optional[float] = None
    needs_escalation: bool = False

class TextResponse(BaseModel):
    answer: str
    sources: List[Source]
    evidence_path: List[EvidenceLink] = []
    latency: LatencyBreakdown
    guardrail: GuardrailResult
    query_id: str
    cached: bool

class Chunk(BaseModel):
    chunk_id: str
    text: str
    strategy: Literal["fixed", "sentence", "semantic", "lang_aware"]
    language: str
    doc_id: str
    token_count: int
    char_start: int
    char_end: int
    metadata: dict
