from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"

    # Chunking
    chunk_size: int = 512        # characters, not tokens
    chunk_overlap: int = 64      # ~12.5% overlap

    # Retrieval
    top_k_results: int = 5
    similarity_threshold: float = 0.75

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Storage
    upload_dir: Path = Path("data/uploads")
    faiss_index_dir: Path = Path("data/faiss_index")

    # Limits
    max_file_size_mb: int = 20
    rate_limit_upload: str = "5/minute"
    rate_limit_query: str = "20/minute"

    class Config:
        env_file = ".env"


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.faiss_index_dir.mkdir(parents=True, exist_ok=True)