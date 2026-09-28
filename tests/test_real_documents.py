"""Integration tests that parse the real PDFs in data/raw (no embedding model needed)."""

from pathlib import Path

import pytest

from challansaathi.chunking import chunk_document, find_headings
from challansaathi.config import PROJECT_ROOT, Settings
from challansaathi.index import load_index, load_or_build_index
from challansaathi.ingest import load_pdf

DATA = PROJECT_ROOT / "data/raw"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="source PDFs not present")


@pytest.fixture(scope="module")
def act():
    return load_pdf(DATA / "MOTOR_VEHICLES.pdf")


def test_act_sections_are_detected_in_order(act):
    numbers = [h.number for h in find_headings(act.text)]
    assert numbers[:4] == ["1", "2", "2A", "3"]
    assert {"129", "163A", "185", "192A", "217A"} <= set(numbers)
    as_ints = [int("".join(ch for ch in n if ch.isdigit())) for n in numbers]
    assert as_ints == sorted(as_ints)


def test_act_chunk_metadata(act):
    chunks = chunk_document(act)
    section_185 = next(c for c in chunks if c.number == "185")
    assert section_185.heading.startswith("Driving by a drunken person")
    assert section_185.unit == "Section"
    assert section_185.state == "India"
    assert section_185.chapter == "CHAPTER XIII"
    assert "blood" in section_185.text
    assert 1 <= section_185.page_start <= section_185.page_end <= len(act.pages)


def test_haryana_rule_138_page_numbers():
    chunks = chunk_document(load_pdf(DATA / "HARYANA.pdf"))
    rule = next(c for c in chunks if c.number == "138")
    assert rule.heading.startswith("Limit of seating capacity")
    assert (rule.page_start, rule.page_end) == (49, 50)
    assert rule.state == "Haryana"


def test_index_is_built_once_then_loaded(tmp_path: Path, embedder):
    settings = Settings(data_dir=DATA, index_dir=tmp_path / "index", embedding_model="fake")
    built = load_or_build_index(settings, embedder)
    assert {c.state for c in built.chunks} == {"India", "Haryana", "Uttar Pradesh"}
    assert built.vectors.ntotal == len(built.chunks)

    loaded = load_index(settings)  # second run: load from disk, no rebuild
    assert loaded.manifest["built_at"] == built.manifest["built_at"]
    assert [c.id for c in loaded.chunks] == [c.id for c in built.chunks]
