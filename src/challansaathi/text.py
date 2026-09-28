"""Tokenisation for BM25 and state detection for English / Hindi-in-Latin (Hinglish) queries."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    # English
    "a an and are as at be by can do does for from has have how i if in is it its me my of on "
    "or shall should that the their there this to under was what when where which who will "
    "with without you your any such said".split()
    # Hinglish
    + "mein me main ka ke ki kya hai hain ho hota hoti hote to se par pe aur ya koi kaun "
    "kitna kitni kitne kab kaise kyun kyu agar nahi na bhi liye wala wali wale karna kare "
    "karte kiya raha rahi rahe tha thi the ko hi sab batao bataye".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


_STATE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Haryana", re.compile(r"\bharyana\b", re.IGNORECASE)),
    ("Uttar Pradesh", re.compile(r"\buttar\s+pradesh\b", re.IGNORECASE)),
    # "UP" / "U.P." written as an abbreviation (case-sensitive: "up" is an English word).
    ("Uttar Pradesh", re.compile(r"\bU\.?P\b")),
    # Lower-case "up" only when followed by a Hinglish postposition: "up mein", "up ka".
    ("Uttar Pradesh", re.compile(r"\bup\s+(?:mein|me|ka|ke|ki|rto)\b", re.IGNORECASE)),
]


def detect_state(query: str) -> str | None:
    """Return the state a query is about, or None when it names no known state."""
    for state, pattern in _STATE_PATTERNS:
        if pattern.search(query):
            return state
    return None


# "section 129", "sec. 185", "s. 194", "dhara 185" (Hindi: धारा) → Section;
# "rule 138", "niyam 12" (नियम) → Rule.
_REFERENCE_RE = re.compile(
    r"\b(?P<unit>sections?|secs?\.?|s\.|dhara|rules?|niyam)\s*(?P<number>\d{1,3}[A-Za-z]{0,2})\b",
    re.IGNORECASE,
)


def parse_references(query: str) -> list[tuple[str, str]]:
    """Explicit provision references in a query, as ``(unit, number)`` pairs.

    >>> parse_references("What does Section 129 say? Also rule 138.")
    [('Section', '129'), ('Rule', '138')]
    """
    references = []
    for match in _REFERENCE_RE.finditer(query):
        word = match["unit"].lower()
        unit = "Rule" if word.startswith(("rule", "niyam")) else "Section"
        references.append((unit, match["number"].upper()))
    return references


# Everyday / Hinglish words → the wording the statutes actually use. Queries say "drunk",
# "daru" or "helmet"; the Act says "drunken person" and "protective headgear".
_GLOSSARY: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), expansion)
    for p, expansion in [
        (r"\b(drunk|drink|drinking|daru|sharab|alcohol|intoxicated)\b", "drunken alcohol blood"),
        (r"\bhelmets?\b", "protective headgear"),
        (r"\bseat\s*belts?\b", "safety belt"),
        (r"\b(over\s*speed\w*|speeding|tez)\b", "excessive speed limits"),
        (r"\b(challans?|jurmana)\b", "fine penalty offence"),
        (r"\blicen[cs]e\b", "licence"),
        (r"\b(mobile|phone|cell\s*phone)\b", "hand-held communication device"),
        (r"\b(red\s*light|signal\s*jump\w*)\b", "traffic signals"),
        (r"\b(puc|pollution)\b", "pollution under control certificate emission"),
        (r"\b(rc|registration\s*certificate)\b", "certificate of registration"),
        (r"\b(triple\s*riding|three\s*on\s*bike)\b", "pillion rider motor cycle"),
        (r"\b(minor|underage|nabalig)\b", "juvenile age limit"),
        (r"\b(bus|buses)\b", "stage carriage public service vehicle"),
        (r"\b(taxi|cab)\b", "motor cab contract carriage"),
        (r"\b(truck|lorry)\b", "goods carriage"),
        (r"\b(accident|takkar)\b", "accident injury"),
        (r"\b(bima|insurance)\b", "insurance policy third party"),
        (r"\b(gaadi|gadi)\b", "motor vehicle"),
    ]
]


def expand_query(query: str) -> str:
    """Append statutory synonyms for everyday terms found in the query."""
    extra = [expansion for pattern, expansion in _GLOSSARY if pattern.search(query)]
    return f"{query} {' '.join(extra)}" if extra else query
