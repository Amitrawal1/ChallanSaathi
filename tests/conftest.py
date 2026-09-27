from __future__ import annotations

import zlib

import numpy as np
import pytest

from challansaathi.chunking import Chunk
from challansaathi.text import tokenize


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder, so tests need no model download."""

    dim = 64

    def encode(self, texts, normalize_embeddings=True, **_):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, zlib.crc32(token.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def make_chunk(document: str, number: str, heading: str, text: str, state: str, part: int = 1):
    return Chunk(
        id=f"{document}:{number}:{part}",
        text=text,
        document=document,
        title=document.upper(),
        state=state,
        document_type="Rules",
        unit="Rule",
        number=number,
        heading=heading,
        chapter="CHAPTER I",
        page_start=1,
        page_end=1,
        part=part,
    )


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def chunks():
    return [
        make_chunk(
            "haryana.pdf",
            "138",
            "Limit of seating capacity",
            "No stage carriage shall carry passengers beyond seating capacity.",
            "Haryana",
        ),
        make_chunk(
            "up.pdf",
            "40",
            "Seating capacity",
            "Seating capacity of buses in Uttar Pradesh shall be fixed.",
            "Uttar Pradesh",
        ),
        make_chunk(
            "act.pdf",
            "185",
            "Driving by a drunken person",
            "Whoever drives with alcohol exceeding 30 mg per 100 ml blood shall be punished.",
            "India",
        ),
        make_chunk(
            "act.pdf",
            "129",
            "Wearing of protective headgear",
            "Every person driving a motor cycle shall wear protective headgear helmet.",
            "India",
        ),
        make_chunk(
            "act.pdf",
            "129",
            "Wearing of protective headgear",
            "The helmet shall conform to BIS standards.",
            "India",
            part=2,
        ),
    ]
