"""Build, save and load the search index (FAISS vectors + chunk store + manifest)."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk, chunk_document
from .config import Settings
from .ingest import load_all

logger = logging.getLogger(__name__)

FAISS_FILE = "vectors.faiss"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


class IndexNotFoundError(FileNotFoundError):
    pass


@dataclass
class SearchIndex:
    chunks: list[Chunk]
    vectors: faiss.Index
    manifest: dict


def load_embedder(model_name: str):
    # Imported lazily: sentence-transformers pulls in torch, which is slow to import.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(embedder, texts: list[str], show_progress: bool = False) -> np.ndarray:
    vectors = embedder.encode(
        texts, normalize_embeddings=True, show_progress_bar=show_progress, batch_size=64
    )
    return np.asarray(vectors, dtype="float32")


def _fingerprint(data_dir: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in sorted(data_dir.glob("*.pdf"))
    }


def build_index(settings: Settings, embedder=None) -> SearchIndex:
    documents = load_all(settings.data_dir)
    chunks: list[Chunk] = []
    for doc in documents:
        doc_chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
        logger.info("%s → %d chunks", doc.filename, len(doc_chunks))
        chunks.extend(doc_chunks)

    embedder = embedder or load_embedder(settings.embedding_model)
    vectors = embed(embedder, [c.embedding_text() for c in chunks], show_progress=True)
    # Inner product on L2-normalised vectors == cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    manifest = {
        "embedding_model": settings.embedding_model,
        "dimension": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "sources": _fingerprint(settings.data_dir),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_index(SearchIndex(chunks, index, manifest), settings.index_dir)
    return SearchIndex(chunks, index, manifest)


def save_index(search_index: SearchIndex, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(search_index.vectors, str(index_dir / FAISS_FILE))
    with open(index_dir / CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in search_index.chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    (index_dir / MANIFEST_FILE).write_text(json.dumps(search_index.manifest, indent=2))
    logger.info("Saved %d chunks to %s", len(search_index.chunks), index_dir)


def load_index(settings: Settings) -> SearchIndex:
    index_dir = settings.index_dir
    if not (index_dir / MANIFEST_FILE).exists():
        raise IndexNotFoundError(
            f"No index in {index_dir}. Build it first with: challansaathi build-index"
        )
    manifest = json.loads((index_dir / MANIFEST_FILE).read_text())
    if manifest["embedding_model"] != settings.embedding_model:
        raise ValueError(
            f"Index was built with {manifest['embedding_model']!r} but settings use "
            f"{settings.embedding_model!r}. Rebuild with: challansaathi build-index"
        )
    if settings.data_dir.exists() and manifest["sources"] != _fingerprint(settings.data_dir):
        logger.warning(
            "PDFs in %s changed since the index was built; rebuild it.", settings.data_dir
        )

    with open(index_dir / CHUNKS_FILE, encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    vectors = faiss.read_index(str(index_dir / FAISS_FILE))
    return SearchIndex(chunks, vectors, manifest)
