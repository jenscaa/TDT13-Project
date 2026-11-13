"""
LangExtract-based relation extractor for Re-DocRED.

Uses Google's LangExtract library for structured, grounded extraction.
Supports Gemini (default) and Ollama backends.

Usage:
  python langextract_extractor.py --data_dir data/re_docred --split dev \
      --backend ollama --model llama3.1:8b --output_dir results/langextract
"""

from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import Document, Entity, load_split, get_relation_stats

try:
    import langextract as lx
    from langextract.data import Extraction as LxExtraction, ExampleData
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    print("WARNING: langextract not installed. Run: pip install langextract")


# Few-shot example builder

def build_examples(train_docs: list[Document], n_docs: int = 5) -> list:
    """
    Build LangExtract few-shot examples from Re-DocRED training documents.

    Each example shows the library a document and the expected relation triples,
    with the entity names extracted verbatim from the document text.
    """
    examples = []
    for doc in train_docs:
        if not doc.labels or len(examples) >= n_docs:
            continue

        text = doc.full_text()
        truncated_text = text[:1500]
        # Build extraction annotations: each relation triple as a dict
        extractions = []
        for rel in doc.labels:
            head_name = doc.entities[rel.head].name
            tail_name = doc.entities[rel.tail].name
            # LangExtract wants extraction_text to appear verbatim in source text
            # Only include if head entity appears in the truncated text
            if head_name in truncated_text:
                extractions.append(
                    LxExtraction(
                        extraction_class="relation",
                        extraction_text=head_name,
                        attributes={
                            "head_entity": head_name,
                            "tail_entity": tail_name,
                            "relation_type": rel.relation,
                        },
                    )
                )

        if extractions:
            examples.append(
                ExampleData(
                    text=truncated_text,
                    extractions=extractions,
                )
            )

    return examples


# Extraction

EXTRACTION_DESCRIPTION = (
    "Extract all relation triples between named entities in the text. "
    "For each relation, identify the head entity (subject), the tail entity (object), "
    "and the relation type. Use only relation types from the provided list. "
    "Only extract relations that are explicitly supported by the text. "
    "Return results as a JSON object with key 'extractions', where each item has fields: "
    "extraction_text (the head entity name as it appears verbatim in the text), "
    "head_entity, tail_entity, and relation_type (must be one of the allowed types)."
)

def build_language_model_params(backend: str, model: str) -> dict:
    """Build params dict passed to extract_document."""
    if backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Get a free key at https://aistudio.google.com"
            )
        return {"model_id": model, "api_key": api_key}

    elif backend == "ollama":
        # LangExtract has native Ollama support
        return {"model_id": model}

    elif backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return {"model_id": model}

    else:
        raise ValueError(f"Unknown backend: {backend}")


def extract_document(
    doc: Document,
    examples: list,
    relation_types: list[str],
    language_model_params: dict,
    retries: int = 2,
) -> list[tuple[str, int, int, str]]:
    """
    Extract relations from one document using LangExtract.
    Returns (doc_title, head_entity_idx, tail_entity_idx, relation) triples.
    """
    text = doc.full_text()
    # Append the allowed relation types to the description to constrain outputs
    rel_hint = "Allowed relation types: " + ", ".join(relation_types[:40])
    description = EXTRACTION_DESCRIPTION + "\n" + rel_hint

    model_id = language_model_params.get("model_id", "gemini-2.0-flash-exp")
    extra_lm_params = {k: v for k, v in language_model_params.items() if k != "model_id"}

    for attempt in range(retries):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    lx.extract,
                    text,
                    prompt_description=description,
                    examples=examples,
                    model_id=model_id,
                    **({"language_model_params": extra_lm_params} if extra_lm_params else {}),
                    extraction_passes=2,
                )
                result = future.result(timeout=180)
            break
        except concurrent.futures.TimeoutError:
            print(f"  [LangExtract timeout attempt {attempt+1}]: exceeded 180s, skipping")
            time.sleep(3)
        except Exception as e:
            print(f"  [LangExtract error attempt {attempt+1}]: {e}")
            time.sleep(3)
    else:
        return []

    # Parse extractions into (title, h_idx, t_idx, relation) triples
    valid_rels = set(relation_types)
    triples = []
    for extraction in result.extractions:
        attrs = extraction.attributes or {}
        head_name = attrs.get("head_entity", "")
        tail_name = attrs.get("tail_entity", "")
        relation = attrs.get("relation_type", "")

        if not head_name or not tail_name or relation not in valid_rels:
            continue

        h_idx = match_entity_index(head_name, doc)
        t_idx = match_entity_index(tail_name, doc)
        if h_idx is not None and t_idx is not None and h_idx != t_idx:
            triples.append((doc.title, h_idx, t_idx, relation))

    return triples


def match_entity_index(name: str, doc: Document) -> int | None:
    """Find entity index by mention name (case-insensitive, with partial fallback)."""
    name_lower = name.lower().strip()
    for entity in doc.entities:
        for mention in entity.mentions:
            if mention.name.lower() == name_lower:
                return entity.idx
    for entity in doc.entities:
        for mention in entity.mentions:
            m = mention.name.lower()
            if name_lower in m or m in name_lower:
                return entity.idx
    return None


# Entry point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/re_docred")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--backend", default="gemini", choices=["gemini", "ollama", "openai"])
    parser.add_argument("--model", default="gemini-2.0-flash-exp",
                        help="Model name for the chosen backend")
    parser.add_argument("--output_dir", default="results/langextract")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Limit number of documents (useful for quick testing)")
    parser.add_argument("--n_examples", type=int, default=5,
                        help="Number of few-shot training examples to include")
    args = parser.parse_args()

    if not LANGEXTRACT_AVAILABLE:
        print("Install langextract first: pip install langextract")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load relation types from BERT run, or derive from training data
    rel2id_path = Path("results/bert/rel2id.json")
    if rel2id_path.exists():
        with open(rel2id_path) as f:
            rel2id = json.load(f)
        relation_types = [r for r in rel2id if r != "Na"]
    else:
        train_docs_tmp = load_split(Path(args.data_dir) / "train_revised.json")
        relation_types = list(get_relation_stats(train_docs_tmp).keys())

    print(f"Relation types: {len(relation_types)}")

    # Load training docs for few-shot examples
    print(f"Building {args.n_examples} few-shot examples from training data...")
    train_docs = load_split(Path(args.data_dir) / "train_revised.json")
    examples = build_examples(train_docs, n_docs=args.n_examples)
    print(f"Built {len(examples)} examples")

    # Load eval split
    docs = load_split(Path(args.data_dir) / f"{args.split}_revised.json")
    if args.max_docs:
        docs = docs[:args.max_docs]
    print(f"Extracting from {len(docs)} documents using {args.backend}/{args.model}...")

    language_model_params = build_language_model_params(args.backend, args.model)

    all_predictions: list[tuple[str, int, int, str]] = []
    errors = 0
    for doc in tqdm(docs, desc=f"LangExtract ({args.backend})"):
        try:
            preds = extract_document(
                doc,
                examples=examples,
                relation_types=relation_types,
                language_model_params=language_model_params,
            )
            all_predictions.extend(preds)
        except Exception as e:
            print(f"  [skip {doc.title}]: {e}")
            errors += 1

    out_name = f"{args.split}_{args.backend}_{args.model.replace(':', '-').replace('/', '-')}_predictions.json"
    out_path = Path(args.output_dir) / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([list(p) for p in all_predictions], f, indent=2)

    print(f"\nSaved {len(all_predictions)} predictions to {out_path}")
    if errors:
        print(f"Errors on {errors} documents")


if __name__ == "__main__":
    main()
