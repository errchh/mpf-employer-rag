from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    project_root: Path = Path(__file__).parent.parent

    zvec_path: Path = project_root / "data" / "zvec"
    documents_path: Path = (
        project_root / "rag" / "doc" / "employers_handbook_mpf_obligations.md"
    )

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "anthropic/claude-3.5-sonnet"

    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8080

    search_top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
