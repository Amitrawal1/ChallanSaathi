"""Structure-aware chunking: one chunk per Section/Rule, sub-split only when too long."""

from __future__ import annotations

import bisect
import re
from dataclasses import asdict, dataclass

from .ingest import LoadedDocument, PageSpan
from .sources import get_source_info

# A provision heading looks like "185. Driving by a drunken person.—" (Acts, CMVR, UP)
# or "138. Limit of seating capacity. [Section 96].--" (Haryana). Requiring the trailing
# dash rules out table-of-contents lines, numbered lists and most footnotes. The title may
# wrap across lines but may not run into the next "N." item. Inserted provisions carry an
# amendment footnote marker, e.g. "2[2A. e-cart and e-rickshaw.—".
HEADING_RE = re.compile(
    r"(?:(?<=\s)|^)(?:\d{1,3}\[)?(?P<num>\d{1,3})(?P<suffix>[A-Z]{0,2})\s?\.\s+"
    r"(?P<title>[A-Za-z\"'\[](?:(?!\n\s*\d{1,3}[A-Z]{0,2}\s?\.\s)[^—–]){1,250}?)"
    r"\s*\.?\s*(?:—|–|--|\.-|:-)"
)
FOOTNOTE_RE = re.compile(r"^(Ins|Subs|Omitted|Added|Rep|The words|Vide|Notification)\b")
CHAPTER_RE = re.compile(r"\bCHAPTER\s+[IVXLC]+[A-Z]?\b")
# Provisions are numbered in increasing order; a larger jump than this is a false match.
MAX_NUMBER_JUMP = 15
# Schedules and forms follow the last provision, e.g. "1[THE FIRST SCHEDULE", "FORM SR-2".
APPENDIX_RE = re.compile(
    r"(?:\d{1,2}\[)?(?P<name>(?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH"
    r"|NINTH|TENTH)\s+SCHEDULE|FORM\s+[A-Z]{0,3}-?\s?\d+[A-Z]?)\b"
)


@dataclass
class Heading:
    start: int
    number: str
    title: str


@dataclass
class Chunk:
    id: str
    text: str
    document: str
    title: str
    state: str
    document_type: str
    unit: str
    number: str | None
    heading: str | None
    chapter: str | None
    page_start: int
    page_end: int
    part: int = 1
    total_parts: int = 1

    @property
    def label(self) -> str:
        """``Section 185`` / ``Rule 138``, or the schedule/form name for appendix chunks."""
        if self.number is None:
            return (self.heading or "Schedule").split(":")[0]
        return f"{self.unit} {self.number}"

    @property
    def citation(self) -> str:
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        return f"{self.title}, {self.label} ({pages})"

    def embedding_text(self) -> str:
        """Text used for search: the heading is prepended so every sub-chunk keeps its context."""
        heading = f"{self.label}: {self.heading}" if self.heading else self.label
        return f"{self.title} | {self.state} | {heading}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def find_headings(text: str) -> list[Heading]:
    """Return the provision headings of a document, in order.

    Every regex match is a candidate. False candidates (sub-rules, cross-references,
    footnotes) break the numbering, so we keep the longest chain of candidates whose
    numbers increase in steps of at most ``MAX_NUMBER_JUMP`` (``2`` → ``2A`` counts as an
    increase). A greedy scan would let one false "32." hide the real rules 25-31.
    """
    candidates: list[tuple[tuple[int, str], Heading]] = []
    for match in HEADING_RE.finditer(text):
        title = " ".join(match["title"].split())
        if FOOTNOTE_RE.match(title):
            continue
        key = (int(match["num"]), match["suffix"])
        candidates.append((key, Heading(match.start(), f"{key[0]}{key[1]}", title)))

    # Longest-chain dynamic programme, O(n^2) over a few hundred candidates.
    best_len = [1] * len(candidates)
    previous: list[int | None] = [None] * len(candidates)
    for j, (key_j, _) in enumerate(candidates):
        if key_j[0] > MAX_NUMBER_JUMP:
            best_len[j] = 0  # a chain must start near the beginning of the numbering
        for i in range(j):
            key_i = candidates[i][0]
            if (
                best_len[i]
                and key_i < key_j
                and key_j[0] - key_i[0] <= MAX_NUMBER_JUMP
                and best_len[i] + 1 > best_len[j]
            ):
                best_len[j], previous[j] = best_len[i] + 1, i
    if not candidates or max(best_len) == 0:
        return []
    cursor: int | None = max(range(len(candidates)), key=best_len.__getitem__)
    chain: list[Heading] = []
    while cursor is not None:
        chain.append(candidates[cursor][1])
        cursor = previous[cursor]
    return chain[::-1]


def _appendix_spans(text: str, start: int, end: int) -> tuple[int, list[tuple[int, int, str]]]:
    """Split schedules/forms off the end of the last provision.

    Returns the new end of the last provision and ``(start, end, heading)`` per appendix.
    """
    markers = list(APPENDIX_RE.finditer(text, start, end))
    if not markers:
        return end, []
    spans = []
    for i, marker in enumerate(markers):
        span_end = markers[i + 1].start() if i + 1 < len(markers) else end
        name = " ".join(marker["name"].split())
        # "THE FIRST SCHEDULE" -> "The First Schedule"; "FORM SR-2" -> "Form SR-2".
        name = f"Form {name[5:]}" if name.startswith("FORM") else name.title()
        # The first line after the marker that is not a "(See rule ...)" reference is its title.
        lines = [ln.strip(" []") for ln in text[marker.end() : span_end].splitlines()]
        title = next((ln for ln in lines if len(ln) > 3 and not ln.lower().startswith("see")), "")
        heading = f"{name}: {title[:80]}" if title else name
        spans.append((marker.start(), span_end, heading))
    return markers[0].start(), spans


# Split long provisions at the most natural boundary available, in this order.
_SEPARATORS = ("\n\n", "\n", ". ", "; ", " ")


def _atomic_units(text: str, size: int, separators: tuple[str, ...] = _SEPARATORS) -> list[str]:
    """Break ``text`` into pieces of at most ``size`` chars, keeping separators attached."""
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        if sep in text:
            parts = text.split(sep)
            parts = [p + sep for p in parts[:-1]] + [parts[-1]]
            units: list[str] = []
            for part in parts:
                units.extend(_atomic_units(part, size, separators[i + 1 :]) if part else [])
            return units
    return [text[i : i + size] for i in range(0, len(text), size)]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` chars, overlapping by up to
    ``chunk_overlap`` chars, preferring paragraph, line and sentence boundaries.

    >>> split_text("One. Two. Three.", chunk_size=12, chunk_overlap=5)
    ['One. Two.', 'Two. Three.']
    """
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in _atomic_units(text, chunk_size):
        if current and length + len(unit) > chunk_size:
            chunks.append("".join(current).strip())
            # Carry the tail of the previous chunk over as overlap, if it leaves room.
            overlap: list[str] = []
            overlap_len = 0
            for previous in reversed(current):
                if overlap_len + len(previous) > chunk_overlap:
                    break
                overlap.insert(0, previous)
                overlap_len += len(previous)
            if overlap_len + len(unit) > chunk_size:
                overlap, overlap_len = [], 0
            current, length = overlap, overlap_len
        current.append(unit)
        length += len(unit)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def _page_range(pages: list[PageSpan], start: int, end: int) -> tuple[int, int]:
    ends = [p.end for p in pages]
    first = bisect.bisect_right(ends, start)
    last = bisect.bisect_left(ends, max(start, end - 1) + 1)
    first = min(first, len(pages) - 1)
    last = min(last, len(pages) - 1)
    return pages[first].page, pages[last].page


def _chapter_lookup(text: str):
    chapters = [(m.start(), m.group(0)) for m in CHAPTER_RE.finditer(text)]
    starts = [pos for pos, _ in chapters]

    def chapter_at(position: int) -> str | None:
        i = bisect.bisect_right(starts, position) - 1
        return chapters[i][1] if i >= 0 else None

    return chapter_at


def chunk_document(
    doc: LoadedDocument, chunk_size: int = 1200, chunk_overlap: int = 150
) -> list[Chunk]:
    info = get_source_info(doc.filename)
    chapter_at = _chapter_lookup(doc.text)
    headings = find_headings(doc.text)

    # (start, end, heading, appendix name) spans. Text before the first heading (title page,
    # notification, table of contents) is not indexed: it only repeats provision titles.
    spans: list[tuple[int, int, Heading | None, str | None]] = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start if i + 1 < len(headings) else len(doc.text)
        spans.append((heading.start, end, heading, None))
    if spans:
        start, end, heading, _ = spans[-1]
        new_end, appendices = _appendix_spans(doc.text, start, end)
        spans[-1] = (start, new_end, heading, None)
        spans.extend((a_start, a_end, None, name) for a_start, a_end, name in appendices)

    chunks: list[Chunk] = []
    for start, end, heading, appendix in spans:
        body = _clean(doc.text[start:end])
        if len(body) < 40:  # page numbers, stray headers
            continue
        page_start, page_end = _page_range(doc.pages, start, end)
        pieces = split_text(body, chunk_size, chunk_overlap)
        for part, piece in enumerate(pieces, start=1):
            number = heading.number if heading else None
            chunks.append(
                Chunk(
                    id=f"{doc.filename}:{number or 'appendix'}:{part}:{start}",
                    text=piece,
                    document=doc.filename,
                    title=info.title,
                    state=info.state,
                    document_type=info.document_type,
                    unit=info.unit,
                    number=number,
                    heading=heading.title if heading else appendix,
                    chapter=chapter_at(start),
                    page_start=page_start,
                    page_end=page_end,
                    part=part,
                    total_parts=len(pieces),
                )
            )
    return chunks
