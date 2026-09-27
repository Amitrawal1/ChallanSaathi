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
