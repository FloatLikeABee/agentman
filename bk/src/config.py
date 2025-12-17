from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    # Database settings
    chroma_persist_directory: str = "./chroma_db"
    data_directory: str = "./data"

    # LLM Provider settings
    default_llm_provider: str = "gemini"  # Options: gemini, qwen, glm
    default_model: str = "gemini-2.5-flash"

    # Gemini settings
    gemini_api_key: str = "AIzaSyAt19tBj232GyyUbM95MlZzZarqZcTKmsc"
    gemini_default_model: str = "gemini-2.5-flash"

    # Qwen settings
    qwen_api_key: str = "sk-fc88e8c463e94a43bc41f1094a28fa1f"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_default_model: str = "qwen3-max"

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug_mode: bool = True
    api_timeout: int = 120  # Timeout in seconds for LLM API calls

    # CORS settings - allow all by default
    # You can override this via environment variable if needed.
    cors_origins: List[str] = ["*"]

    # Email settings
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None

    # Financial API settings
    alpha_vantage_api_key: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()