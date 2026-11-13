"""LLM-based relation extractor using Ollama for Re-DocRED.

Uses LangExtract if it is installed, and otherwise falls back to prompting
the Ollama API directly.

Usage:
    python llm_extractor.py --data_dir data/re_docred --split dev \
        --model llama3.1:8b --output_dir results/llm --mode zero_shot
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
from data_loader import Document, load_split

try:
    import langextract  # noqa: F401
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False

try:
    import ollama as ollama_client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from google import genai as genai_client
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# Prompt builders

SYSTEM_PROMPT = (
    "Extract relations between named entities from the given text. "
    "Output ONLY a JSON array of relation triples. Each triple has keys: "
    '"head" (entity name), "tail" (entity name), "relation" (relation type as a short label). '
    "If no relations are found, output an empty array []."
)


def build_zero_shot_prompt(doc: Document, relation_types: list[str]) -> str:
    text = doc.full_text()
    entity_names = list({m.name for e in doc.entities for m in e.mentions})
    rel_list = "\n".join(f"  - {r}" for r in relation_types)
    return (
        f"Text:\n{text}\n\n"
        f"Entities present: {', '.join(entity_names)}\n\n"
        f"Possible relation types (use these exact labels):\n{rel_list}\n\n"
        "Extract all relations between the listed entities. "
        "Output a JSON array of objects with keys: head, tail, relation."
    )


def build_few_shot_prompt(
    doc: Document,
    relation_types: list[str],
    examples: list[dict],
) -> str:
    zero_shot = build_zero_shot_prompt(doc, relation_types)
    example_str = ""
    for ex in examples[:3]:
        example_str += (
            f"\n\nExample text: {ex['text']}\n"
            f"Example output: {json.dumps(ex['output'])}"
        )
    return f"Here are examples of the task:{example_str}\n\nNow do the same:\n\n{zero_shot}"


# Ollama backend

def call_ollama(model: str, system: str, user: str, retries: int = 3, timeout: int = 90) -> str:
    """Call Ollama and return the response text."""
    for attempt in range(retries):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    ollama_client.chat,
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    options={"temperature": 0.0, "num_predict": 512},
                )
                response = future.result(timeout=timeout)
            return response["message"]["content"]
        except concurrent.futures.TimeoutError:
            print(f"  [Ollama timeout attempt {attempt+1}]: exceeded {timeout}s, skipping")
            time.sleep(2)
        except Exception as e:
            print(f"  [Ollama error attempt {attempt+1}]: {e}")
            time.sleep(2)
    return "[]"


# Gemini backend

def call_gemini(model: str, system: str, user: str, retries: int = 5) -> str:
    """Call Gemini API and return the response text."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    client = genai_client.Client(api_key=api_key)
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=genai_client.types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
            time.sleep(13)  # stay within free tier 5 RPM limit
            return response.text
        except Exception as e:
            err = str(e)
            wait = 60 if "503" in err else 30 if "429" in err else 10
            print(f"  [Gemini error attempt {attempt+1}]: {e}")
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)
    return "[]"


# LangExtract backend

def call_langextract(model: str, doc: Document, relation_types: list[str]) -> list[dict]:
    """Extract relation triples using LangExtract's local Ollama provider."""
    if not LANGEXTRACT_AVAILABLE:
        raise ImportError("LangExtract is not installed. Run: pip install langextract")
    text = doc.full_text()
    if not text.strip() or not relation_types:
        return []

    entity_names = sorted({m.name for e in doc.entities for m in e.mentions})
    prompt = (
        "Extract all directed relations between the listed entities. "
        "Use extraction_class 'relation' and copy supporting text verbatim "
        "into extraction_text. Include attributes 'head', 'tail', and 'relation'. "
        "Head and tail must be listed entity names; relation must be an exact "
        "allowed label. Extract no relations without evidence.\n"
        f"Entities: {json.dumps(entity_names)}\n"
        f"Allowed relation labels: {json.dumps(relation_types)}"
    )
    examples = [
        langextract.data.ExampleData(
            text="Alice is related to Bob.",
            extractions=[
                langextract.data.Extraction(
                    extraction_class="relation",
                    extraction_text="Alice is related to Bob.",
                    attributes={
                        "head": "Alice",
                        "tail": "Bob",
                        "relation": relation_types[0],
                    },
                )
            ],
        )
    ]
    result = langextract.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id=model,
        model_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        temperature=0.0,
    )
    triples = []
    for extraction in result.extractions or []:
        attributes = extraction.attributes or {}
        if extraction.extraction_class == "relation" and all(
            isinstance(attributes.get(key), str) for key in ("head", "tail", "relation")
        ):
            triples.append({key: attributes[key] for key in ("head", "tail", "relation")})
    return triples


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_triples(raw: str) -> list[dict]:
    """Parse LLM output into a list of {head, tail, relation} dicts."""
    # Find JSON array in the output
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        triples = json.loads(raw[start:end])
        if isinstance(triples, list):
            return [t for t in triples if isinstance(t, dict) and "head" in t and "tail" in t and "relation" in t]
    except json.JSONDecodeError:
        pass
    return []


def match_entity_index(name: str, doc: Document) -> int | None:
    """Find entity index by mention name (case-insensitive)."""
    name_lower = name.lower()
    for entity in doc.entities:
        for mention in entity.mentions:
            if mention.name.lower() == name_lower:
                return entity.idx
    # Partial match fallback
    for entity in doc.entities:
        for mention in entity.mentions:
            if name_lower in mention.name.lower() or mention.name.lower() in name_lower:
                return entity.idx
    return None


# Main extraction loop

def extract_document(
    doc: Document,
    model: str,
    relation_types: list[str],
    mode: str,
    few_shot_examples: list[dict] | None = None,
    use_langextract: bool = False,
    backend: str = "ollama",
) -> list[tuple[str, int, int, str]]:
    """
    Extract relations for a single document.
    Returns list of (doc_title, head_entity_idx, tail_entity_idx, relation) triples.
    """
    if use_langextract and LANGEXTRACT_AVAILABLE:
        raw_triples = call_langextract(model, doc, relation_types)
    else:
        if mode == "few_shot" and few_shot_examples:
            prompt = build_few_shot_prompt(doc, relation_types, few_shot_examples)
        else:
            prompt = build_zero_shot_prompt(doc, relation_types)

        if backend == "gemini":
            raw = call_gemini(model, SYSTEM_PROMPT, prompt)
        else:
            raw = call_ollama(model, SYSTEM_PROMPT, prompt)
        raw_triples = parse_triples(raw)

    # Map entity names back to indices
    result = []
    valid_rels = set(relation_types)
    for triple in raw_triples:
        rel = triple.get("relation", "")
        if rel not in valid_rels:
            continue
        h_idx = match_entity_index(triple.get("head", ""), doc)
        t_idx = match_entity_index(triple.get("tail", ""), doc)
        if h_idx is not None and t_idx is not None and h_idx != t_idx:
            result.append((doc.title, h_idx, t_idx, rel))

    return result


def build_few_shot_examples(train_docs: list[Document], n: int = 3) -> list[dict]:
    """Extract simple few-shot examples from training data."""
    examples = []
    for doc in train_docs[:20]:
        if not doc.labels:
            continue
        text = doc.full_text()[:500]  # truncate for prompt length
        output = [
            {
                "head": doc.entities[rel.head].name,
                "tail": doc.entities[rel.tail].name,
                "relation": rel.relation,
            }
            for rel in doc.labels
        ]
        examples.append({"text": text, "output": output})
        if len(examples) >= n:
            break
    return examples


# Entry point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/re_docred")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "gemini"])
    parser.add_argument("--output_dir", default="results/llm")
    parser.add_argument("--mode", default="zero_shot", choices=["zero_shot", "few_shot"])
    parser.add_argument("--max_docs", type=int, default=None, help="Limit number of docs (for testing)")
    parser.add_argument("--n_examples", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--use_langextract", action="store_true")
    args = parser.parse_args()

    if args.backend == "gemini" and not GEMINI_AVAILABLE:
        print("ERROR: google-generativeai not installed. Run: pip install google-generativeai")
        sys.exit(1)
    if args.backend == "ollama" and not OLLAMA_AVAILABLE:
        print("ERROR: ollama not installed. Run: pip install ollama")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load relation types from BERT run (or from train data)
    rel2id_path = Path("results/bert/rel2id.json")
    if rel2id_path.exists():
        with open(rel2id_path) as f:
            rel2id = json.load(f)
        relation_types = [r for r in rel2id if r != "Na"]
    else:
        train_docs = load_split(Path(args.data_dir) / "train_revised.json")
        from data_loader import get_relation_stats
        relation_types = list(get_relation_stats(train_docs).keys())

    print(f"Relation types: {len(relation_types)}")

    docs = load_split(Path(args.data_dir) / f"{args.split}_revised.json")
    if args.max_docs:
        docs = docs[:args.max_docs]
    print(f"Documents: {len(docs)}")

    few_shot_examples = None
    if args.mode == "few_shot":
        train_docs = load_split(Path(args.data_dir) / "train_revised.json")
        few_shot_examples = build_few_shot_examples(train_docs, n=args.n_examples)
        print(f"Few-shot examples: {len(few_shot_examples)}")

    all_predictions: list[tuple[str, int, int, str]] = []
    for doc in tqdm(docs, desc=f"Extracting ({args.mode})"):
        preds = extract_document(
            doc,
            model=args.model,
            relation_types=relation_types,
            mode=args.mode,
            few_shot_examples=few_shot_examples,
            use_langextract=args.use_langextract,
            backend=args.backend,
        )
        all_predictions.extend(preds)

    model_safe = args.model.replace(":", "-").replace("/", "-")
    out_path = Path(args.output_dir) / f"{args.split}_{args.backend}_{model_safe}_{args.mode}_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([list(p) for p in all_predictions], f, indent=2)
    print(f"Saved {len(all_predictions)} predictions to {out_path}")


if __name__ == "__main__":
    main()
