import pytest

from challansaathi.sources import get_source_info
from challansaathi.text import detect_state, tokenize


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
