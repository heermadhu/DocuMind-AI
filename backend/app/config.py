from pathlib import Path
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "DocuMind AI"
    DATABASE_URL: str = "sqlite:///./data/documind.db"
    UPLOAD_DIR: str = "./data/uploads"
    STORAGE_DIR: str = "./data/storage"
    VECTOR_STORE_PATH: str = "./data/vector_store"
    MAX_UPLOAD_SIZE: int = 20 * 1024 * 1024
    MAX_PDF_PAGES: int = 200
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 80
    OCR_ENABLED: bool = True
    TESSERACT_CMD: str | None = None
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_REVISION: str = "main"
    RETRIEVAL_MODE: str = "hybrid"
    DENSE_RELEVANCE_THRESHOLD: float = 0.15
    LLM_PROVIDER: str = "disabled"
    LLM_API_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_TIMEOUT: float = 30.0
    LLM_MAX_CONTEXT_CHARS: int = 12000
    LLM_TEMPERATURE: float = 0.0
    LLM_API_KEY: str | None = None
    LLM_MODEL: str = "gpt-4o-mini"
    VISION_PROVIDER: str = "disabled"
    VISION_API_BASE_URL: str = "https://api.openai.com/v1"
    VISION_MODEL: str = "gpt-4o-mini"
    VISION_API_TIMEOUT: float = 45.0
    VISION_API_KEY: str | None = None
    CORS_ORIGINS: str = "*"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

if settings.CHUNK_SIZE < 1:
    raise ValueError("CHUNK_SIZE must be positive.")
if settings.CHUNK_OVERLAP < 0 or settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be between zero and CHUNK_SIZE - 1.")
if settings.RETRIEVAL_MODE not in {"dense", "hybrid"}:
    raise ValueError("RETRIEVAL_MODE must be 'dense' or 'hybrid'.")
if not -1.0 <= settings.DENSE_RELEVANCE_THRESHOLD <= 1.0:
    raise ValueError("DENSE_RELEVANCE_THRESHOLD must be between -1 and 1.")
if settings.LLM_API_TIMEOUT <= 0 or settings.VISION_API_TIMEOUT <= 0:
    raise ValueError("Provider request timeouts must be positive.")
if settings.LLM_MAX_CONTEXT_CHARS < 1:
    raise ValueError("LLM_MAX_CONTEXT_CHARS must be positive.")
cors_origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
if "*" in cors_origins and len(cors_origins) > 1:
    raise ValueError("CORS_ORIGINS cannot combine wildcard and explicit origins.")


def provider_url_is_local(value: str) -> bool:
    return urlparse(value).hostname in {"localhost", "127.0.0.1", "::1"}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
settings.UPLOAD_DIR = str((PROJECT_ROOT / settings.UPLOAD_DIR).resolve())
settings.STORAGE_DIR = str((PROJECT_ROOT / settings.STORAGE_DIR).resolve())
settings.VECTOR_STORE_PATH = str((PROJECT_ROOT / settings.VECTOR_STORE_PATH).resolve())
settings.DATABASE_URL = settings.DATABASE_URL.replace("sqlite:///./", f"sqlite:///{PROJECT_ROOT}/")
