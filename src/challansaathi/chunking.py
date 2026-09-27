"""Structure-aware chunking: one chunk per Section/Rule, sub-split only when too long."""

from __future__ import annotations

import bisect
import re
from dataclasses import asdict, dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .ingest import LoadedDocument, PageSpan
from .sources import get_source_info

# A provision heading looks like "185. Driving by a drunken person.—" (Acts, CMVR, UP)
# or "138. Limit of seating capacity. [Section 96].--" (Haryana). Requiring the trailing
# dash rules out table-of-contents lines, numbered lists and most footnotes. The title may
# wrap across lines but may not run into the next "N." item. Inserted provisions carry an
# amendment footnote marker, e.g. "2[2A. e-cart and e-rickshaw.—".
HEADING_RE = re.compile(
    r"(?:(?<=\s)|^)(?:\d{1,3}\[)?(?P<num>\d{1,3})(?P<suffix>[A-Z]{0,2})\s?\.\s+"
    r"(?P<title>[A-Za-z\"'(\[](?:(?!\n\s*\d{1,3}[A-Z]{0,2}\s?\.\s)[^—–]){1,250}?)"
    r"\s*\.?\s*(?:—|–|--|\.-|:-)"
)
FOOTNOTE_RE = re.compile(r"^(Ins|Subs|Omitted|Added|Rep|The words|Vide|Notification)\b")
CHAPTER_RE = re.compile(r"\bCHAPTER\s+[IVXLC]+[A-Z]?\b")
# Provisions are numbered in increasing order; a larger jump than this is a false match.
MAX_NUMBER_JUMP = 15


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
        if self.number is None:
            return "Preamble"
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
    headings: list[Heading] = []
    last = 0
    for match in HEADING_RE.finditer(text):
        number, suffix = int(match["num"]), match["suffix"]
        title = " ".join(match["title"].split())
        if FOOTNOTE_RE.match(title):
            continue
        # Accept "2A" after "2", or the next number within a plausible jump.
        if last < number <= last + MAX_NUMBER_JUMP or (number == last and suffix):
            headings.append(Heading(match.start(), f"{number}{suffix}", title))
            last = number
    return headings


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
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chapter_at = _chapter_lookup(doc.text)
    headings = find_headings(doc.text)

    # (start, end, heading) spans; text before the first heading is the preamble.
    spans: list[tuple[int, int, Heading | None]] = []
    first_start = headings[0].start if headings else len(doc.text)
    if first_start > 0:
        spans.append((0, first_start, None))
    for i, heading in enumerate(headings):
        end = headings[i + 1].start if i + 1 < len(headings) else len(doc.text)
        spans.append((heading.start, end, heading))

    chunks: list[Chunk] = []
    for start, end, heading in spans:
        body = _clean(doc.text[start:end])
        if len(body) < 40:  # page numbers, stray headers
            continue
        page_start, page_end = _page_range(doc.pages, start, end)
        pieces = [body] if len(body) <= chunk_size else splitter.split_text(body)
        for part, piece in enumerate(pieces, start=1):
            number = heading.number if heading else None
            chunks.append(
                Chunk(
                    id=f"{doc.filename}:{number or 'preamble'}:{part}:{start}",
                    text=piece,
                    document=doc.filename,
                    title=info.title,
                    state=info.state,
                    document_type=info.document_type,
                    unit=info.unit,
                    number=number,
                    heading=heading.title if heading else None,
                    chapter=chapter_at(start),
                    page_start=page_start,
                    page_end=page_end,
                    part=part,
                    total_parts=len(pieces),
                )
            )
    return chunks
