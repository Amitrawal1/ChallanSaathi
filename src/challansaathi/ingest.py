"""PDF loading: one continuous text per document plus a character-offset → page map."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class PageSpan:
    start: int
    end: int
    page: int  # 1-based physical page number


@dataclass
class LoadedDocument:
    filename: str
    text: str
    pages: list[PageSpan]


def load_pdf(path: Path) -> LoadedDocument:
    reader = PdfReader(path)
    parts: list[str] = []
    pages: list[PageSpan] = []
    offset = 0
    for number, page in enumerate(reader.pages, start=1):
        content = (page.extract_text() or "") + "\n"
        parts.append(content)
        pages.append(PageSpan(offset, offset + len(content), number))
        offset += len(content)
    logger.info("Loaded %s (%d pages)", path.name, len(pages))
    return LoadedDocument(path.name, "".join(parts), pages)


def load_all(data_dir: Path) -> list[LoadedDocument]:
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not paths:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")
    return [load_pdf(p) for p in paths]
