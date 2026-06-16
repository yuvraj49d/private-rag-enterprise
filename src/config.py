import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Using lightweight open-source models
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    LOCAL_LLM_MODEL: str = "qwen2.5:1.5b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # RAG Hyperparameters
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Local storage paths
    DATA_DIR: str = "data/"
    DB_DIR: str = "vectorstore/db"

    class Config:
        env_file = ".env"

settings = Settings()