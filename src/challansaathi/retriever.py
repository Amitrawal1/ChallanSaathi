"""Hybrid retrieval: FAISS vectors + BM25, fused with weighted Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .config import Settings
from .index import SearchIndex, embed
from .sources import CENTRAL
from .text import tokenize

# Long provisions are split into many sub-chunks; cap how many go to the LLM per provision.
MAX_PARTS_PER_RESULT = 3


@dataclass
class SearchResult:
    """One provision. ``parts`` holds every matched sub-chunk of it, in document order."""

    chunk: Chunk
    score: float
    vector_rank: int | None
    bm25_rank: int | None
    parts: list[Chunk] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n[...]\n".join(p.text for p in self.parts or [self.chunk])


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], weights: Sequence[float], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ranked id lists: score(id) = sum(weight / (k + rank)). Best first."""
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    def __init__(self, index: SearchIndex, embedder, settings: Settings):
        self.chunks = index.chunks
        self.vectors = index.vectors
        self.embedder = embedder
        self.settings = settings
        self.bm25 = BM25Okapi([tokenize(c.embedding_text()) for c in self.chunks])
        self._states = np.array([c.state for c in self.chunks])

    def _allowed_ids(self, state: str | None) -> np.ndarray | None:
        """A state question searches that state's rules plus central law; None means everything."""
        if state is None:
            return None
        return np.flatnonzero(np.isin(self._states, [state, CENTRAL]))

    def _vector_ranking(self, query: str, allowed: np.ndarray | None, k: int) -> list[int]:
        query_vector = embed(self.embedder, [query])
        params = None
        if allowed is not None:
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed.astype("int64")))
        _, ids = self.vectors.search(query_vector, k, params=params)
        return [int(i) for i in ids[0] if i != -1]

    def _bm25_ranking(self, query: str, allowed: np.ndarray | None, k: int) -> list[int]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if allowed is not None:
            mask = np.full(len(scores), -np.inf)
            mask[allowed] = 0.0
            scores = scores + mask
        top = np.argsort(scores)[::-1][:k]
        return [int(i) for i in top if scores[i] > 0]

    def search(
        self, query: str, state: str | None = None, top_k: int | None = None
    ) -> list[SearchResult]:
        s = self.settings
        top_k = top_k or s.top_k
        allowed = self._allowed_ids(state)
        vector_ids = self._vector_ranking(query, allowed, s.candidates_per_retriever)
        bm25_ids = self._bm25_ranking(query, allowed, s.candidates_per_retriever)
        fused = reciprocal_rank_fusion(
            [vector_ids, bm25_ids], [s.vector_weight, s.bm25_weight], k=s.rrf_k
        )
        vector_rank = {i: r for r, i in enumerate(vector_ids, start=1)}
        bm25_rank = {i: r for r, i in enumerate(bm25_ids, start=1)}

        # Group sub-chunks of the same provision so each result is a distinct Section/Rule.
        grouped: dict[tuple[str, str | None, int], SearchResult] = {}
        for i, score in fused:
            chunk = self.chunks[i]
            key = (chunk.document, chunk.number, chunk.page_start)
            if key in grouped:
                if len(grouped[key].parts) < MAX_PARTS_PER_RESULT:
                    grouped[key].parts.append(chunk)
                continue
            if len(grouped) == top_k:
                continue
            grouped[key] = SearchResult(chunk, score, vector_rank.get(i), bm25_rank.get(i), [chunk])
        for result in grouped.values():
            result.parts.sort(key=lambda c: c.part)
        return list(grouped.values())
