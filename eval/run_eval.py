"""Retrieval evaluation: Hit@1, Hit@5 and MRR@5 for each retrieval configuration.

Usage: python eval/run_eval.py [--questions eval/questions.jsonl] [--out eval/results.md]

A question counts as a hit when any of its ``relevant`` (document, number) provisions is
retrieved. Questions go through the same state detection as the app.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from challansaathi.config import get_settings
from challansaathi.index import load_embedder, load_index
from challansaathi.retriever import HybridRetriever
from challansaathi.text import detect_state

EVAL_DIR = Path(__file__).resolve().parent

# (label, retriever mode, query expansion)
CONFIGS = [
    ("Vector only", "vector", "off"),
    ("BM25 only", "bm25", "off"),
    ("BM25 only + query expansion", "bm25", "bm25"),
    ("Hybrid (RRF)", "hybrid", "off"),
    ("Hybrid + expansion (BM25 side)", "hybrid", "bm25"),
    ("Hybrid + expansion (both sides)", "hybrid", "all"),
]


def evaluate(retriever: HybridRetriever, questions: list[dict], mode: str, k: int = 5) -> dict:
    hits1 = hitsk = reciprocal = refused = 0
    misses = []
    for q in questions:
        relevant = {tuple(r) for r in q["relevant"]}
        results = retriever.search(
            q["question"], state=detect_state(q["question"]), top_k=k, mode=mode
        )
        if not results:
            refused += 1
        ranks = [
            rank
            for rank, r in enumerate(results, start=1)
            if (r.chunk.document, r.chunk.number) in relevant
        ]
        if ranks:
            hits1 += ranks[0] == 1
            hitsk += 1
            reciprocal += 1 / ranks[0]
        else:
            misses.append(q["id"])
    n = len(questions)
    return {
        "hit@1": hits1 / n,
        f"hit@{k}": hitsk / n,
        f"mrr@{k}": reciprocal / n,
        "refused": refused,
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results.md")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line]
    settings = get_settings()
    index = load_index(settings)
    embedder = load_embedder(settings.embedding_model)
    retriever = HybridRetriever(index, embedder, settings)

    rows = []
    for label, mode, expansion in CONFIGS:
        retriever.settings = replace(settings, query_expansion=expansion)
        metrics = evaluate(retriever, questions, mode)
        rows.append((label, metrics))
        print(f"{label:34} {json.dumps({k: v for k, v in metrics.items() if k != 'misses'})}")

    lines = [
        f"Retrieval evaluation on {len(questions)} questions "
        f"({index.manifest['chunk_count']} chunks, top-5, state auto-detected).",
        "",
        "| Configuration | Hit@1 | Hit@5 | MRR@5 | Missed question ids |",
        "|---|---|---|---|---|",
    ]
    for label, m in rows:
        lines.append(
            f"| {label} | {m['hit@1']:.1%} | {m['hit@5']:.1%} | {m['mrr@5']:.3f} | "
            f"{', '.join(map(str, m['misses'])) or '-'} |"
        )
    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
