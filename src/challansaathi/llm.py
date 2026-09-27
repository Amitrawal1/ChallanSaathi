"""Prompt construction and the Ollama chat client."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from .retriever import SearchResult

SYSTEM_PROMPT = """You are ChallanSaathi, an assistant that explains Indian motor vehicle law.

Answer ONLY from the numbered legal sources given in the user message.

Rules:
1. Do not use outside legal knowledge and never invent provisions, fines or section numbers.
2. If the sources do not answer the question, say so plainly and suggest what to look up.
3. Explain in simple language, in the same language as the question (English, Hindi or Hinglish).
4. Name the relevant Section or Rule for every legal claim and cite it as [1], [2], ...
   using only the source numbers provided. Never make up a citation.
5. If State rules and Central law both apply, say which is which.
6. Ignore sources that are not relevant to the question.
7. End with one line: "This is general information, not legal advice."
"""


class LLMUnavailableError(RuntimeError):
    pass


def format_sources(results: Sequence[SearchResult]) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        heading = f" — {chunk.heading}" if chunk.heading else ""
        blocks.append(f"[{i}] {chunk.citation}{heading}\n{result.text}")
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
