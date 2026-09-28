"""Prompt construction and the Ollama chat client."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from .retriever import SearchResult
from .sources import CENTRAL, get_source_info

SYSTEM_PROMPT = """You are ChallanSaathi, an assistant that explains Indian motor vehicle law.

You receive numbered LEGAL SOURCES and a QUESTION. Answer ONLY from those sources.

Rules:
1. Never use outside knowledge. Never invent provisions, fines, amounts or section numbers.
   If a fact is not written in the sources, do not state it.
2. If the sources do not answer the question, reply that the loaded documents do not cover it
   and stop. Do not answer from memory.
3. Start with the main rule that answers the question, then any exceptions.
4. After every sentence that states a legal rule, put the source number in square brackets,
   e.g. "The driver must wear a helmet [2]." Use only the numbers given.
5. Name the Section or Rule you rely on (e.g. "Section 129 of the Motor Vehicles Act").
6. Each source is labelled Central law or State law. When you use a State rule, say which
   State it belongs to; when you use Central law, say it applies across India.
7. Each source shows "Text as of"; when you state a fine or amount, say it is as per the text
   of that date and may have been amended since.
8. Use simple words, in the same language as the question (English, Hindi or Hinglish).
9. Describe what the law says; do not advise the user what to do in their own case.
10. Ignore sources that are not relevant. Keep the answer short: at most 8 sentences."""


class LLMUnavailableError(RuntimeError):
    pass


def format_sources(results: Sequence[SearchResult]) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        heading = f" — {chunk.heading}" if chunk.heading else ""
        info = get_source_info(chunk.document)
        scope = "Central law" if chunk.state == CENTRAL else f"State law: {chunk.state}"
        blocks.append(
            f"[{i}] {chunk.citation}{heading}\n"
            f"({scope}. Text as of: {info.text_as_of})\n{result.text}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, results: Sequence[SearchResult]) -> list[dict[str, str]]:
    user = f"LEGAL SOURCES:\n\n{format_sources(results)}\n\nQUESTION: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class OllamaLLM:
    def __init__(self, model: str, host: str):
        import ollama

        self.model = model
        self.client = ollama.Client(host=host)

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        import httpx
        import ollama

        try:
            for part in self.client.chat(
                model=self.model, messages=messages, stream=True, options={"temperature": 0.1}
            ):
                yield part["message"]["content"]
        except (ConnectionError, httpx.ConnectError) as exc:
            raise LLMUnavailableError(
                "Cannot reach Ollama. Install it from https://ollama.com and run `ollama serve`."
            ) from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise LLMUnavailableError(
                    f"Model {self.model!r} is not pulled. Run: ollama pull {self.model}"
                ) from exc
            raise
