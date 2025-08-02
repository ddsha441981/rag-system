from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    session_id: Optional[str] = None
    search_mode: Optional[str] = "hybrid"

class QueryResponse(BaseModel):
    question: str
    answer: str
    verification: Optional[str] = None
    context_chunks: int
    sources: List[str]
    tokens_used: int
    success: bool
    cached: bool = False
    search_mode: str

class DocumentResponse(BaseModel):
    filename: str
    chunks_processed: int
    success: bool
    error: Optional[str] = None

class SystemStats(BaseModel):
    documents_indexed: int
    cache_available: bool
    bm25_index_size: int
    active_sessions: int
