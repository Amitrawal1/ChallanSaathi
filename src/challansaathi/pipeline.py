"""High-level question answering: detect state → retrieve → prompt → generate."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

from .config import Settings, get_settings
from .index import load_embedder, load_index, load_or_build_index
from .llm import OllamaLLM, build_messages
from .retriever import HybridRetriever, SearchResult
from .sources import CENTRAL, get_source_info
from .text import detect_state

DISCLAIMER = (
    "*ChallanSaathi gives general information from the cited legal texts, not legal advice. "
    "Laws and fines are amended often; check the official text or a lawyer for your case.*"
)
NO_SOURCES_ANSWER = (
    "I could not find a relevant provision in the loaded motor vehicle law documents for this "
    "question, so I can't answer it. ChallanSaathi covers the Motor Vehicles Act 1988, the "
    "Central Motor Vehicles Rules 1989 and the Haryana and Uttar Pradesh motor vehicle rules."
)


def source_to_dict(
    result: SearchResult, source_id: int, excerpt_chars: int = 500
) -> dict[str, Any]:
    """JSON-serialisable description of one retrieved provision."""
    chunk = result.chunk
    return {
        "source_id": source_id,
        "citation": chunk.citation,
        "title": chunk.title,
        "document": chunk.document,
        "jurisdiction": "Central" if chunk.state == CENTRAL else chunk.state,
        "unit": chunk.unit,
        "number": chunk.number,
        "heading": chunk.heading,
        "chapter": chunk.chapter,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "text_as_of": get_source_info(chunk.document).text_as_of,
        "score": round(result.score, 5),
        "excerpt": result.text[:excerpt_chars],
    }


@dataclass
class Answer:
    question: str
    state: str | None
    sources: list[SearchResult]
    stream: Iterator[str] = field(repr=False)
    _text: str | None = field(default=None, repr=False)

    def text(self) -> str:
        """Consume the stream (once) and return the full answer."""
        if self._text is None:
            self._text = "".join(self.stream)
        return self._text

    def to_dict(self) -> dict[str, Any]:
        """``{"question", "state", "answer", "sources"}`` — generates the answer if needed."""
        return {
            "question": self.question,
            "state": self.state,
            "answer": self.text(),
            "sources": [source_to_dict(r, i) for i, r in enumerate(self.sources, start=1)],
        }


class ChallanSaathi:
    """The assistant. Loads (or, if ``auto_build``, builds) the index once at start-up."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm=None,
        embedder=None,
        auto_build: bool = True,
    ):
        self.settings = settings or get_settings()
        embedder = embedder or load_embedder(self.settings.embedding_model)
        index = (
            load_or_build_index(self.settings, embedder)
            if auto_build
            else load_index(self.settings)
        )
        self.retriever = HybridRetriever(index, embedder, self.settings)
        self.llm = llm or OllamaLLM(self.settings.ollama_model, self.settings.ollama_host)

    def retrieve(
        self, question: str, state: str | None = None
    ) -> tuple[str | None, list[SearchResult]]:
        """Detect the state (unless given) and return the matching provisions."""
        state = state or detect_state(question)
        return state, self.retriever.search(question, state=state)

    def ask(self, question: str, state: str | None = None) -> Answer:
        """Retrieve sources and return an :class:`Answer` whose text streams from the LLM."""
        state, results = self.retrieve(question, state)
        if not results:
            return Answer(question, state, [], iter([NO_SOURCES_ANSWER]))
        stream = self.llm.stream(build_messages(question, results))
        # The disclaimer is appended here, not left to the model, so it is always present.
        return Answer(question, state, results, chain(stream, ["\n\n" + DISCLAIMER]))
