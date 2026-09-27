"""High-level question answering: retrieve → prompt → generate."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .index import load_embedder, load_index
from .llm import OllamaLLM, build_messages
from .retriever import HybridRetriever, SearchResult
from .text import detect_state

NO_SOURCES_ANSWER = (
    "I could not find a relevant provision in the loaded documents for this question."
)


@dataclass
class Answer:
    question: str
    state: str | None
    sources: list[SearchResult]
    stream: Iterator[str] = field(repr=False)

    def text(self) -> str:
        """Consume the stream and return the full answer."""
        return "".join(self.stream)


class ChallanSaathi:
    def __init__(self, settings: Settings | None = None, llm=None, embedder=None):
        self.settings = settings or get_settings()
        index = load_index(self.settings)
        embedder = embedder or load_embedder(self.settings.embedding_model)
        self.retriever = HybridRetriever(index, embedder, self.settings)
        self.llm = llm or OllamaLLM(self.settings.ollama_model, self.settings.ollama_host)

    def retrieve(
        self, question: str, state: str | None = None
    ) -> tuple[str | None, list[SearchResult]]:
        state = state or detect_state(question)
        return state, self.retriever.search(question, state=state)

    def ask(self, question: str, state: str | None = None) -> Answer:
        state, results = self.retrieve(question, state)
        if not results:
            return Answer(question, state, [], iter([NO_SOURCES_ANSWER]))
        return Answer(question, state, results, self.llm.stream(build_messages(question, results)))
