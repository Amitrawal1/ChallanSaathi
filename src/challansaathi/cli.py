"""Command-line interface: ``challansaathi build-index | search | ask``."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import get_settings
from .index import build_index
from .llm import LLMUnavailableError
from .sources import known_states


def _print_sources(results) -> None:
    for i, r in enumerate(results, start=1):
        c = r.chunk
        print(f"[{i}] {c.citation}" + (f" — {c.heading}" if c.heading else ""))
        similarity = "-" if r.similarity is None else f"{r.similarity:.3f}"
        print(
            f"    score={r.score:.4f} vector_rank={r.vector_rank} bm25_rank={r.bm25_rank} "
            f"cosine={similarity}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="challansaathi", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-index", help="Parse PDFs in data/raw and build the search index")

    for name, help_text in [
        ("search", "Show retrieved sources only"),
        ("ask", "Answer a question"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("question")
        p.add_argument("--state", choices=known_states(), help="Override state detection")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose or args.command == "build-index" else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = get_settings()

    if args.command == "build-index":
        index = build_index(settings)
        print(f"Indexed {len(index.chunks)} chunks → {settings.index_dir}")
        return 0

    from .pipeline import ChallanSaathi

    assistant = ChallanSaathi(settings)
    if args.command == "search":
        state, results = assistant.retrieve(args.question, args.state)
        print(f"State filter: {state or 'none (all documents)'}\n")
        _print_sources(results)
        return 0

    answer = assistant.ask(args.question, args.state)
    try:
        for token in answer.stream:
            print(token, end="", flush=True)
    except LLMUnavailableError as exc:
        print(f"Error: {exc}\n\nRetrieved sources:", file=sys.stderr)
        _print_sources(answer.sources)
        return 1
    print("\n\nSources:")
    _print_sources(answer.sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
