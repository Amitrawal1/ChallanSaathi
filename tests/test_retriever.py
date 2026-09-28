import faiss
import pytest

from challansaathi.config import Settings
from challansaathi.index import SearchIndex, embed
from challansaathi.llm import build_messages
from challansaathi.retriever import HybridRetriever, reciprocal_rank_fusion


def test_rrf_rewards_items_ranked_by_both():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], [1.0, 1.0], k=60)
    assert [item for item, _ in fused][:2] == [1, 3]
    assert {item for item, _ in fused} == {1, 2, 3, 4}


def test_rrf_weights():
    fused = reciprocal_rank_fusion([[1], [2]], [0.9, 0.1])
    assert fused[0][0] == 1


@pytest.fixture
def retriever(chunks, embedder):
    vectors = embed(embedder, [c.embedding_text() for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return HybridRetriever(
        SearchIndex(chunks, index, {}), embedder, Settings(top_k=3, min_similarity=0.0)
    )


def test_state_filter_keeps_state_and_central_law(retriever):
    results = retriever.search("seating capacity", state="Haryana")
    states = {r.chunk.state for r in results}
    assert results[0].chunk.number == "138"
    assert "Uttar Pradesh" not in states


def test_no_state_searches_everything(retriever):
    results = retriever.search("drunken alcohol driving")
    assert results[0].chunk.number == "185"


def test_parts_of_one_provision_are_grouped(retriever):
    results = retriever.search("helmet protective headgear")
    helmet = [r for r in results if r.chunk.number == "129"]
    assert len(helmet) == 1
    assert [p.part for p in helmet[0].parts] == [1, 2]
    assert "BIS standards" in helmet[0].text


def test_mode_selects_single_retriever(retriever):
    vector_only = retriever.search("helmet", mode="vector")
    bm25_only = retriever.search("helmet", mode="bm25")
    assert all(r.vector_rank is not None for r in vector_only)
    assert all(r.bm25_rank is not None for r in bm25_only)
    assert bm25_only[0].chunk.number == "129"


def test_out_of_scope_question_returns_nothing(retriever):
    retriever.settings = Settings(min_similarity=0.4)
    assert retriever.search("chocolate cake recipe") == []
    assert retriever.search("protective headgear helmet motor cycle") != []


def test_explicit_section_reference_ranks_first(retriever):
    results = retriever.search("What does Rule 185 say?")
    assert results[0].chunk.number == "185"
    # A named provision is returned even when the words match nothing else.
    retriever.settings = Settings(min_similarity=0.99)
    assert retriever.search("rule 129")[0].chunk.number == "129"


def test_prompt_numbers_sources(retriever):
    results = retriever.search("helmet protective headgear")
    messages = build_messages("helmet?", results)
    assert messages[0]["role"] == "system"
    assert "[1] " in messages[1]["content"]
    assert "QUESTION: helmet?" in messages[1]["content"]
