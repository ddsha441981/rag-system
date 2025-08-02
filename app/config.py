# Replace your app/config.py with this version:
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Optional, Union


class Settings(BaseSettings):
    # API Keys
    groq_api_key: str
    openai_api_key: Optional[str] = None

    # Database
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "enhanced_knowledge_base"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl_hours: int = 24

    # Models
    default_llm_model: str = "llama3-8b-8192"
    embedding_model: str = "all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Search
    default_search_k: int = 5
    hybrid_search_alpha: float = 0.7
    max_chunk_size: int = 500
    chunk_overlap: int = 50

    # Application
    max_upload_size_mb: int = 50
    allowed_file_types: Union[List[str], str] = "pdf,docx,txt,md,csv,xlsx,html,json"
    log_level: str = "INFO"

    @field_validator('allowed_file_types')
    @classmethod
    def parse_file_types(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',') if item.strip()]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
