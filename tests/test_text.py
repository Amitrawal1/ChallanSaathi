import pytest

from challansaathi.sources import get_source_info
from challansaathi.text import detect_state, expand_query, parse_references, tokenize


def test_tokenize_drops_english_and_hinglish_stopwords():
    assert tokenize("Haryana mein seating capacity ka rule kya hai?") == [
        "haryana",
        "seating",
        "capacity",
        "rule",
    ]


@pytest.mark.parametrize(
    ("query", "state"),
    [
        ("Haryana mein seating capacity ka rule kya hai?", "Haryana"),
        ("helmet rules in Uttar Pradesh", "Uttar Pradesh"),
        ("UP mein challan kitna hai", "Uttar Pradesh"),
        ("fine in U.P. for no helmet", "Uttar Pradesh"),
        ("up mein challan kitna hai", "Uttar Pradesh"),
        ("What happens if I pick up a passenger?", None),
        ("fine for drunk driving", None),
    ],
)
def test_detect_state(query, state):
    assert detect_state(query) == state


def test_source_registry():
    assert get_source_info("MOTOR_VEHICLES.pdf").unit == "Section"
    assert get_source_info("UP.pdf").state == "Uttar Pradesh"
    assert get_source_info("new_doc.pdf").state == "India"


@pytest.mark.parametrize(
    ("query", "references"),
    [
        ("What does Section 129 of the Act say?", [("Section", "129")]),
        ("sec. 185 aur rule 138 batao", [("Section", "185"), ("Rule", "138")]),
        ("dhara 163a kya hai", [("Section", "163A")]),
        ("UP niyam 12", [("Rule", "12")]),
        ("fine for 2 wheelers", []),
    ],
)
def test_parse_references(query, references):
    assert parse_references(query) == references


def test_expand_query_adds_statutory_terms():
    expanded = expand_query("daru pee ke gaadi chalana")
    assert "drunken" in expanded and "motor vehicle" in expanded
    assert expand_query("helmet fine").startswith("helmet fine ")
    assert "protective headgear" in expand_query("helmet fine")
    assert expand_query("transfer of ownership") == "transfer of ownership"
