import faiss
import pytest

from challansaathi.config import Settings
from challansaathi.index import IndexNotFoundError, SearchIndex, embed, save_index
from challansaathi.pipeline import ChallanSaathi


class FakeLLM:
    def __init__(self):
        self.messages = None

    def stream(self, messages):
        self.messages = messages
        yield "Answer "
        yield "[1]"


@pytest.fixture
def settings(tmp_path, chunks, embedder):
    settings = Settings(
        data_dir=tmp_path / "raw", index_dir=tmp_path / "index", embedding_model="fake"
    )
    vectors = embed(embedder, [c.embedding_text() for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    save_index(
        SearchIndex(chunks, index, {"embedding_model": "fake", "sources": {}}), settings.index_dir
    )
    return settings


def test_ask_detects_state_and_streams_answer(settings, embedder):
    llm = FakeLLM()
    assistant = ChallanSaathi(settings, llm=llm, embedder=embedder)
    answer = assistant.ask("Haryana mein seating capacity ka rule kya hai?")

    assert answer.state == "Haryana"
    assert answer.text() == "Answer [1]"
    assert answer.sources[0].chunk.number == "138"
    assert "Limit of seating capacity" in llm.messages[1]["content"]


def test_missing_index_has_helpful_error(tmp_path, embedder):
    settings = Settings(index_dir=tmp_path / "missing", embedding_model="fake")
    with pytest.raises(IndexNotFoundError, match="build-index"):
        ChallanSaathi(settings, llm=FakeLLM(), embedder=embedder)


def test_embedding_model_mismatch_is_rejected(settings, embedder):
    other = Settings(data_dir=settings.data_dir, index_dir=settings.index_dir, embedding_model="x")
    with pytest.raises(ValueError, match="Rebuild"):
        ChallanSaathi(other, llm=FakeLLM(), embedder=embedder)
