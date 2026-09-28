from challansaathi.chunking import chunk_document, find_headings, split_text
from challansaathi.ingest import LoadedDocument, PageSpan

ACT_TEXT = """THE MOTOR VEHICLES ACT, 1988
ARRANGEMENT OF SECTIONS
1. Short title, extent and commencement.
2. Definitions.
CHAPTER I
PRELIMINARY
1. Short title, extent and commencement.—(1) This Act may be called the Motor Vehicles Act.
(2) It extends to the whole of India.
2. Definitions.—In this Act, unless the context otherwise requires,— (1) "area" means an area.
1. Ins. by Act 54 of 1988, s. 2 (w.e.f. 14-11-1988).—footnote text here.
2A. e-cart and e-rickshaw.—(1) The provisions of this Act shall apply to e-carts.
CHAPTER II
LICENSING OF DRIVERS
3. Necessity for driving licence.—(1) No person shall drive a motor vehicle in any public place.
"""


def test_find_headings_skips_toc_and_footnotes():
    headings = find_headings(ACT_TEXT)
    assert [h.number for h in headings] == ["1", "2", "2A", "3"]
    assert headings[0].title == "Short title, extent and commencement"
    assert headings[3].title == "Necessity for driving licence"


def test_find_headings_haryana_style():
    text = (
        "1. Seating space. [Section 111].-- Every seat shall be 40 cm wide. "
        "2. Limit of seating capacity. [Section 111(2)(a)].-- No vehicle shall carry more."
    )
    headings = find_headings(text)
    assert [h.number for h in headings] == ["1", "2"]
    assert headings[1].title.startswith("Limit of seating capacity")


def _document(filename: str, text: str) -> LoadedDocument:
    middle = len(text) // 2
    return LoadedDocument(filename, text, [PageSpan(0, middle, 1), PageSpan(middle, len(text), 2)])


def test_chunk_document_metadata():
    chunks = chunk_document(_document("MOTOR_VEHICLES.pdf", ACT_TEXT))
    by_number = {c.number: c for c in chunks}

    # The title page / table of contents before the first provision is not indexed.
    assert None not in by_number
    section_3 = by_number["3"]
    assert section_3.unit == "Section"
    assert section_3.title == "Motor Vehicles Act, 1988"
    assert section_3.state == "India"
    assert section_3.chapter == "CHAPTER II"
    assert section_3.page_start == section_3.page_end == 2
    assert section_3.citation == "Motor Vehicles Act, 1988, Section 3 (p. 2)"
    assert by_number["1"].chapter == "CHAPTER I"
    assert by_number["1"].page_start == 1


def test_false_heading_does_not_hide_following_rules():
    # A cross-reference that looks like rule "32" must not swallow the real rules 25-27.
    text = (
        "23. Alpha.— text. 24. Beta.— see clause 32. Explanation.— more text. "
        "25. Gamma.— text. 26. Delta.— text. 27. Epsilon.— text."
    )
    text = " ".join(f"{n}. Rule {n}.— body." for n in range(1, 23)) + " " + text
    numbers = [h.number for h in find_headings(text)]
    assert numbers[-5:] == ["23", "24", "25", "26", "27"]
    assert "32" not in numbers


def test_sub_rule_is_not_a_heading():
    text = "1. First rule.— (1) text. 2. Second rule.— body (3) In considering.— x. 3. Third.— y."
    assert [h.number for h in find_headings(text)] == ["1", "2", "3"]


def test_schedules_and_forms_are_split_from_last_rule():
    text = (
        "1. Short title.— These rules may be called the test rules and apply everywhere.\n"
        "2. Power to give directions.— The State Government may issue such directions.\n"
        "1[FIRST SCHEDULE\n[See Rule 41(6)]\nLetters allowed to the Registering Authority A B C\n"
        "FORM SR-2\n[See Rule 12(a)]\nTransport Vehicle Driver's Badge details and signature\n"
    )
    chunks = chunk_document(_document("UP.pdf", text))
    labels = [c.label for c in chunks]
    assert labels == ["Rule 1", "Rule 2", "First Schedule", "Form SR-2"]
    assert "SCHEDULE" not in chunks[1].text
    assert chunks[3].heading == "Form SR-2: Transport Vehicle Driver's Badge details and signature"
    assert chunks[3].citation.startswith("Uttar Pradesh Motor Vehicles Rules, 1998, Form SR-2 (")


def test_long_provision_is_split_with_context():
    body = "9. Long rule.—" + " ".join(f"Clause {i} applies to vehicles." for i in range(200))
    chunks = chunk_document(_document("HARYANA.pdf", body), chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    assert {c.number for c in chunks} == {"9"}
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.total_parts == len(chunks) for c in chunks)
    assert all(c.embedding_text().startswith("Haryana Motor Vehicles Rules") for c in chunks)
    assert all("Rule 9: Long rule" in c.embedding_text() for c in chunks)


def test_split_text_respects_size_and_overlaps():
    lines = [f"({i}) Sub-rule {i} says vehicles must comply with clause {i}." for i in range(60)]
    text = "\n".join(lines)
    pieces = split_text(text, chunk_size=300, chunk_overlap=80)
    assert len(pieces) > 1
    assert all(len(p) <= 300 for p in pieces)
    # Consecutive pieces share text, and no sub-rule is lost.
    assert all(a.splitlines()[-1] in b for a, b in zip(pieces, pieces[1:], strict=False))
    assert all(f"({i}) Sub-rule {i} " in "".join(pieces) for i in range(60))


def test_split_text_hard_splits_unbroken_text():
    assert split_text("x" * 25, chunk_size=10, chunk_overlap=0) == ["x" * 10, "x" * 10, "x" * 5]
