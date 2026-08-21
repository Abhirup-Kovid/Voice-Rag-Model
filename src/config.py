"""Configuration module."""
import os
import sys
import structlog
from pydantic_settings import BaseSettings

logger = structlog.get_logger(__name__)

class Settings(BaseSettings):
    SARVAM_API_KEY: str
    GROQ_API_KEY: str
    INDEX_DIR: str = "indexes"
    DATA_DIR: str = "data"
    MODEL_NAME: str = "intfloat/multilingual-e5-small"
    SARVAM_URL: str = "https://api.sarvam.ai/speech-to-text"
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    TOP_K_RETRIEVE: int = 8
    TOP_K_FINAL: int = 3
    CACHE_SIZE: int = 100
    LATENCY_BUDGET_MS: int = 200

    EMBED_MODEL_DIR: str = "models/onnx_e5_small"
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "awaaz-rag"
    PINECONE_NAMESPACE: str = "awaaz"
    PINECONE_REGION: str = "us-east-1"

    class Config:
        env_file = ".env"

try:
    settings = Settings()
    if not settings.SARVAM_API_KEY or not settings.GROQ_API_KEY:
        logger.error("API keys missing from .env")
        sys.exit(1)
except Exception as e:
    logger.error("Failed to load settings", error=str(e))
    sys.exit(1)