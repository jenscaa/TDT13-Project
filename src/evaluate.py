"""Evaluation utilities for document-level relation extraction.

Gold labels and predictions are both lists of (doc_title, head_idx,
tail_idx, relation). Precision, recall and F1 are micro-averaged over all
relation instances.

Usage:
    python evaluate.py <gold_split.json> <predictions.json>
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from data_loader import Document, load_split


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def __str__(self) -> str:
        return (
            f"P={self.precision:.4f}  R={self.recall:.4f}  F1={self.f1:.4f}  "
            f"(TP={self.tp}, FP={self.fp}, FN={self.fn})"
        )


def compute_metrics(
    gold: list[tuple[str, int, int, str]],
    pred: list[tuple[str, int, int, str]],
) -> Metrics:
    """
    Micro-averaged P/R/F1.

    Both gold and pred are lists of (doc_title, head_entity_idx, tail_entity_idx, relation).
    Entity indices are relative to their document's vertexSet.
    """
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def per_relation_metrics(
    gold: list[tuple[str, int, int, str]],
    pred: list[tuple[str, int, int, str]],
) -> dict[str, Metrics]:
    """Compute P/R/F1 per relation type."""
    relations = {r for _, _, _, r in gold} | {r for _, _, _, r in pred}
    result = {}
    for rel in sorted(relations):
        g = [(t, h, ta, r) for t, h, ta, r in gold if r == rel]
        p = [(t, h, ta, r) for t, h, ta, r in pred if r == rel]
        result[rel] = compute_metrics(g, p)
    return result


def gold_from_documents(documents: list[Document]) -> list[tuple[str, int, int, str]]:
    """Extract gold triples from a list of Document objects."""
    triples = []
    for doc in documents:
        for rel in doc.labels:
            triples.append((doc.title, rel.head, rel.tail, rel.relation))
    return triples


def save_predictions(
    predictions: list[tuple[str, int, int, str]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([list(t) for t in predictions], f, indent=2)
    print(f"Saved {len(predictions)} predictions to {path}")


def load_predictions(path: str | Path) -> list[tuple[str, int, int, str]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [tuple(t) for t in raw]


def error_analysis(
    gold: list[tuple[str, int, int, str]],
    pred: list[tuple[str, int, int, str]],
    top_n: int = 10,
) -> None:
    """Print most common false positive and false negative relation types."""
    gold_set = set(gold)
    pred_set = set(pred)

    fp_counts: dict[str, int] = defaultdict(int)
    fn_counts: dict[str, int] = defaultdict(int)

    for t, h, ta, r in pred_set - gold_set:
        fp_counts[r] += 1
    for t, h, ta, r in gold_set - pred_set:
        fn_counts[r] += 1

    print("\n--- Top false positive relation types (hallucinated) ---")
    for rel, cnt in sorted(fp_counts.items(), key=lambda x: -x[1])[:top_n]:
        print(f"  {rel}: {cnt}")

    print("\n--- Top false negative relation types (missed) ---")
    for rel, cnt in sorted(fn_counts.items(), key=lambda x: -x[1])[:top_n]:
        print(f"  {rel}: {cnt}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: evaluate.py <gold_split.json> <predictions.json>")
        sys.exit(1)

    docs = load_split(sys.argv[1])
    gold = gold_from_documents(docs)
    pred = load_predictions(sys.argv[2])

    print(f"Gold: {len(gold)} triples, Pred: {len(pred)} triples")
    metrics = compute_metrics(gold, pred)
    print(f"Overall: {metrics}")

    per_rel = per_relation_metrics(gold, pred)
    print(f"\nPer-relation F1 (top 15 by F1):")
    for rel, m in sorted(per_rel.items(), key=lambda x: -x[1].f1)[:15]:
        print(f"  {rel:40s} F1={m.f1:.3f}  (P={m.precision:.3f}, R={m.recall:.3f})")

    error_analysis(gold, pred)
