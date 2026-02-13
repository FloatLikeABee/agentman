from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
import os


def _read_local_key(filename: str) -> str:
  """
  Read a secret key from a local file (one-line), returning empty string if missing.
  The file is expected to live at the project root and is gitignored.
  """
  try:
    # config.py lives in src/, project root is one level up
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
      return f.read().strip()
  except OSError:
    return ""


class Settings(BaseSettings):
    # Database settings
    chroma_persist_directory: str = "./chroma_db"
    data_directory: str = "./data"

    # LLM Provider settings
    # Options: gemini, qwen, mistral, groq
    default_llm_provider: str = "gemini"
    default_model: str = "gemini-2.5-flash"

    # Gemini settings
    # Default: read from local gitignored file `gemini_key` at project root.
    # Can still be overridden via environment variable GEMINI_API_KEY / .env.
    gemini_api_key: str = _read_local_key("gemini_key")
    gemini_default_model: str = "gemini-2.5-flash"

    # Qwen settings
    qwen_api_key: str = "sk-206d748313fb42dab3910dc3407f441b"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_default_model: str = "qwen3-max"

    # Mistral settings
    mistral_api_key: str = "2IGzr4XnznEjh3O3vs0wFf0lwh7r7yhU"
    mistral_default_model: str = "mistral-large-latest"

    # Groq settings
    # IMPORTANT: set GROQ_API_KEY in your local .env; do not commit real keys.
    groq_api_key: str = ""
    groq_default_model: str = "llama-3.3-70b-versatile"

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_ssl_verify: bool = True  # Set to False to disable SSL verification for HuggingFace downloads (development only)
    hf_download_timeout: int = 300  # Timeout in seconds for HuggingFace model downloads
    hf_proxy: Optional[str] = None  # Proxy URL for HuggingFace downloads (e.g., "http://proxy.example.com:8080" or "https://proxy.example.com:8080")
    hf_mirror: Optional[str] = None  # Mirror endpoint for HuggingFace (e.g., "https://hf-mirror.com" for China mirror)

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug_mode: bool = Field(True, validation_alias="DEBUG")  # .env can use DEBUG=true (or DEBUG_MODE=true)
    api_timeout: int = 180  # Timeout in seconds for LLM API calls (increased for Qwen compatibility)

    # CORS settings - allow all by default
    # You can override these via environment variables if needed.
    # `cors_origins` is kept for compatibility but we primarily rely on `cors_origin_regex`
    # so that the server echoes back the actual Origin instead of "*".
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_origin_regex: str = ".*"

    # Email settings
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None

    # Financial API settings
    alpha_vantage_api_key: Optional[str] = 'ED2OR4DHOZH8Z56J'

    # Web Search API settings (optional - for Tavily API)
    # Tavily has a free tier: 128 searches/month
    # If not set, will use free search engines (unlimited)
    tavily_api_key: Optional[str] = None

    # Text-to-SQL default SQL Server connection (used when no db_tool_id / connection_config / connection_string provided)
    # Override via env: TEXT_TO_SQL_DEFAULT_HOST, TEXT_TO_SQL_DEFAULT_PORT, etc.
    # If password contains $ and you use .env, use single quotes so the shell does not expand it: TEXT_TO_SQL_DEFAULT_PASSWORD='$transfinder2006'
    text_to_sql_default_host: str = "192.168.9.9"
    text_to_sql_default_port: int = 1433
    text_to_sql_default_database: str = "team2_ent"
    text_to_sql_default_username: str = "tfuser"
    text_to_sql_default_password: str = "$transfinder2006"

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra env vars (e.g. DEBUG) so .env from env.example doesn't raise


settings = Settings()
