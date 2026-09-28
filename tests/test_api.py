import pytest
from fastapi.testclient import TestClient

from challansaathi.api import create_app
from challansaathi.llm import LLMUnavailableError
from challansaathi.pipeline import ChallanSaathi

from .test_pipeline import FakeLLM, settings  # noqa: F401  (fixture re-export)


class DownLLM:
    def stream(self, messages):
        raise LLMUnavailableError("Cannot reach Ollama.")
        yield  # pragma: no cover


@pytest.fixture
def make_client(settings, embedder):  # noqa: F811
    def make(llm):
        app = create_app(lambda: ChallanSaathi(settings, llm=llm, embedder=embedder))
        return TestClient(app)

    return make


def test_ask_returns_answer_and_sources(make_client):
    with make_client(FakeLLM()) as client:
        response = client.post("/ask", json={"question": "Haryana seating capacity rule"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Answer [1]")
    assert body["state"] == "Haryana"
    assert body["sources"][0]["number"] == "138"
    assert body["sources"][0]["page_start"] == 1


def test_explicit_state_overrides_detection(make_client):
    with make_client(FakeLLM()) as client:
        body = client.post(
            "/search", json={"question": "helmet protective headgear capacity", "state": "India"}
        )
    assert body.status_code == 200
    assert {s["jurisdiction"] for s in body.json()["sources"]} == {"Central"}


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "   "}, {"question": "helmet", "state": "Goa"}],
)
def test_invalid_requests_are_rejected(make_client, payload):
    with make_client(FakeLLM()) as client:
        assert client.post("/ask", json=payload).status_code == 422


def test_llm_down_returns_503(make_client):
    with make_client(DownLLM()) as client:
        response = client.post("/ask", json={"question": "Haryana seating capacity rule"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_health(make_client):
    with make_client(FakeLLM()) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == 5
