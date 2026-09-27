"""Runtime settings, overridable through ``CHALLANSAATHI_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"CHALLANSAATHI_{name}", default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(_env("DATA_DIR", str(PROJECT_ROOT / "data/raw")))
    )
    index_dir: Path = field(
        default_factory=lambda: Path(_env("INDEX_DIR", str(PROJECT_ROOT / "vectorstore")))
    )
    embedding_model: str = field(
        default_factory=lambda: _env(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.2"))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))

    chunk_size: int = 1200
    chunk_overlap: int = 150

    top_k: int = 5
    candidates_per_retriever: int = 30
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    rrf_k: int = 60


def get_settings() -> Settings:
    return Settings()
