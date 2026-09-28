"""Runtime settings, overridable through ``CHALLANSAATHI_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
    # Append statutory synonyms (text.expand_query) to the query for: nothing ("off"),
    # BM25 only ("bm25") or both retrievers ("all"). Chosen with eval/run_eval.py.
    query_expansion: Literal["off", "bm25", "all"] = "all"
    candidates_per_retriever: int = 30
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    reference_weight: float = 2.0  # explicit "Section 129" / "Rule 138" matches
    rrf_k: int = 60
    # Below this cosine similarity (and with no explicit "Section N" reference) a question is
    # treated as out of scope and the LLM is not called. Calibrated on real queries: unrelated
    # questions (cooking, cricket, GST, programming) score <= 0.30, short in-scope ones
    # ("mobile phone") ~0.35. Borderline questions reach the LLM, which is told to refuse.
    min_similarity: float = 0.30


def get_settings() -> Settings:
    return Settings()
