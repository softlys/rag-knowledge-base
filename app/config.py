import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")
    model: str = os.environ.get("RAG_MODEL", "claude-sonnet-4-6")
    chroma_dir: str = os.environ.get("CHROMA_DIR", str(BASE_DIR / "data" / "chroma"))
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.environ.get("TOP_K", "4"))


settings = Settings()
