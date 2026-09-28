"""REST API. Run with: ``uvicorn challansaathi.api:app`` (or ``make api``)."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from .llm import LLMUnavailableError
from .pipeline import ChallanSaathi, source_to_dict

StateName = Literal["Haryana", "Uttar Pradesh", "India"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["Penalty for drunk driving?"])
    state: StateName | None = Field(
        default=None,
        description="Restrict to a state's rules + central law ('India' = central law only). "
        "Detected from the question when omitted.",
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return value


class Source(BaseModel):
    source_id: int
    citation: str
    title: str
    document: str
    jurisdiction: str
    unit: str
    number: str | None
    heading: str | None
    chapter: str | None
    page_start: int
    page_end: int
    text_as_of: str
    score: float
    excerpt: str


class SearchResponse(BaseModel):
    question: str
    state: str | None
    sources: list[Source]


class AskResponse(SearchResponse):
    answer: str


def create_app(assistant_factory: Callable[[], ChallanSaathi] = ChallanSaathi) -> FastAPI:
    """Build the app; the assistant (index + embedding model) is loaded once at start-up."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.assistant = assistant_factory()
        yield

    app = FastAPI(
        title="ChallanSaathi API",
        description="Indian motor vehicle law Q&A with cited sources. "
        "General information, not legal advice.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def assistant(request: Request) -> ChallanSaathi:
        return request.app.state.assistant

    @app.get("/health")
    def health(request: Request) -> dict:
        a = assistant(request)
        return {
            "status": "ok",
            "chunks": len(a.retriever.chunks),
            "llm_model": a.settings.ollama_model,
        }

    # Sync handlers: FastAPI runs them in a thread pool, so slow LLM calls don't block the loop.
    @app.post("/search", response_model=SearchResponse)
    def search(body: AskRequest, request: Request) -> dict:
        state, results = assistant(request).retrieve(body.question, body.state)
        return {
            "question": body.question,
            "state": state,
            "sources": [source_to_dict(r, i) for i, r in enumerate(results, start=1)],
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> dict:
        try:
            return assistant(request).ask(body.question, body.state).to_dict()
        except LLMUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
