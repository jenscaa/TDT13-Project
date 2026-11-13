"""BERT baseline for document-level relation extraction on Re-DocRED.

Each document is encoded with BERT (truncated to 512 tokens). For each entity
pair (h, t), mention embeddings are averaged into one vector per entity, and
[h; t; h * t] is classified into relation types. The task is multi-label, so
one forward pass predicts all relations for a pair.

Usage:
    python bert.py --data_dir data/re_docred --output_dir results/bert --epochs 10
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# Add src to path when running from project root
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import Document, load_split, get_relation_stats

# Constants
MODEL_NAME = "bert-base-cased"
MAX_SEQ_LEN = 512
NA_RELATION = "Na"  # no-relation label


def build_relation_vocab(documents: list[Document]) -> tuple[dict[str, int], list[str]]:
    """Build relation type -> index mapping (index 0 = Na / no relation)."""
    stats = get_relation_stats(documents)
    rel2id = {NA_RELATION: 0}
    for rel in stats:
        rel2id[rel] = len(rel2id)
    id2rel = [r for r, _ in sorted(rel2id.items(), key=lambda x: x[1])]
    return rel2id, id2rel


# Dataset

class DocREDataset(Dataset):
    """Entity-pair dataset for document-level relation extraction.

    Yields one sample per ordered entity pair (h, t) with a multi-hot label
    vector (index 0 = Na). Documents are tokenized up front, and mentions are
    stored as inclusive subword spans (max 4 per entity). Pairs where an entity
    has no mentions left after truncation to max_len are skipped.

    Args:
        documents: Parsed Re-DocRED documents.
        tokenizer: HuggingFace fast tokenizer.
        rel2id: Relation name to index mapping, with Na at index 0.
        max_len: Maximum sequence length in subwords.
    """

    def __init__(
        self,
        documents: list[Document],
        tokenizer,
        rel2id: dict[str, int],
        max_len: int = MAX_SEQ_LEN,
    ):
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.num_relations = len(rel2id)
        self.samples = []

        for doc in tqdm(documents, desc="Tokenizing", leave=False):
            self._process_doc(doc, max_len)

    def _process_doc(self, doc: Document, max_len: int) -> None:
        # Flatten tokens and track sent/token offsets
        tokens: list[str] = []
        for sent in doc.sents:
            tokens.extend(sent)

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)  # list[int|None]

        # Map flat token index -> first subword index in the encoded sequence
        word_to_subword: dict[int, int] = {}
        for subword_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx not in word_to_subword:
                word_to_subword[word_idx] = subword_idx

        # Build flat token offset: cumulative sent lengths
        sent_offsets = [0]
        for sent in doc.sents:
            sent_offsets.append(sent_offsets[-1] + len(sent))

        # For each entity, collect subword positions of all mentions
        entity_subword_spans: list[list[tuple[int, int]]] = []
        for entity in doc.entities:
            spans = []
            for mention in entity.mentions:
                flat_start = sent_offsets[mention.sent_id] + mention.pos[0]
                flat_end = sent_offsets[mention.sent_id] + mention.pos[1]
                sw_start = word_to_subword.get(flat_start)
                sw_end = word_to_subword.get(flat_end - 1)
                if sw_start is not None and sw_end is not None and sw_start < max_len and sw_end < max_len:
                    spans.append((sw_start, sw_end))
            entity_subword_spans.append(spans)

        # Build gold label dict
        ep_labels: dict[tuple[int, int], list[str]] = doc.entity_pair_labels()

        # Create one sample per entity pair
        n = len(doc.entities)
        for h in range(n):
            for t in range(n):
                if h == t:
                    continue
                h_spans = entity_subword_spans[h]
                t_spans = entity_subword_spans[t]
                if not h_spans or not t_spans:
                    continue

                label_vec = torch.zeros(self.num_relations)
                rels = ep_labels.get((h, t), [])
                for rel in rels:
                    if rel in self.rel2id:
                        label_vec[self.rel2id[rel]] = 1.0
                if not rels:
                    label_vec[0] = 1.0  # Na

                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "h_spans": h_spans[:4],
                    "t_spans": t_spans[:4],
                    "labels": label_vec,
                    "doc_title": doc.title,
                    "h_idx": h,
                    "t_idx": t,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "h_spans": [b["h_spans"] for b in batch],
        "t_spans": [b["t_spans"] for b in batch],
        "labels": torch.stack([b["labels"] for b in batch]),
        "doc_title": [b["doc_title"] for b in batch],
        "h_idx": [b["h_idx"] for b in batch],
        "t_idx": [b["t_idx"] for b in batch],
    }


# Model

class DocREModel(nn.Module):
    """BERT-based entity-pair relation classifier.

    Encodes the document, mean-pools each entity's mention spans into one
    vector, and feeds [h; t; h * t] through a two-layer MLP. Returns one
    logit per relation type, for use with a multi-label (sigmoid) loss.

    Args:
        model_name: HuggingFace model name or path for the encoder.
        num_relations: Number of relation types, including Na.
        hidden_size: Encoder hidden size (768 for BERT-base).
    """

    def __init__(self, model_name: str, num_relations: int, hidden_size: int = 768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_relations),
        )

    def _pool_entity(self, hidden: torch.Tensor, spans: list[list[tuple[int, int]]]) -> torch.Tensor:
        """Average pooling over all mention spans in the batch."""
        batch_reps = []
        for b_spans in spans:
            mention_reps = []
            for start, end in b_spans:
                # Average tokens within [start, end]
                mention_reps.append(hidden[len(batch_reps), start:end + 1].mean(dim=0))
            if mention_reps:
                batch_reps.append(torch.stack(mention_reps).mean(dim=0))
            else:
                batch_reps.append(hidden[len(batch_reps), 0])  # fallback: CLS
        return torch.stack(batch_reps)

    def forward(self, input_ids, attention_mask, h_spans, t_spans):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)

        h_rep = self._pool_entity(hidden, h_spans)
        t_rep = self._pool_entity(hidden, t_spans)

        combined = torch.cat([h_rep, t_rep, h_rep * t_rep], dim=-1)
        logits = self.classifier(combined)
        return logits


# Training

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_docs = load_split(Path(args.data_dir) / "train_revised.json")
    dev_docs = load_split(Path(args.data_dir) / "dev_revised.json")
    if args.max_docs:
        train_docs = train_docs[:args.max_docs]
        dev_docs = dev_docs[:max(1, args.max_docs // 6)]

    rel2id, id2rel = build_relation_vocab(train_docs)
    print(f"Relations: {len(rel2id)} types (including Na)")

    with open(Path(args.output_dir) / "rel2id.json", "w") as f:
        json.dump(rel2id, f, indent=2)

    train_dataset = DocREDataset(train_docs, tokenizer, rel2id)
    dev_dataset = DocREDataset(dev_docs, tokenizer, rel2id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = DocREModel(args.model_name, num_relations=len(rel2id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, batch["h_spans"], batch["t_spans"])
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate on dev
        f1, preds_out = evaluate(model, dev_loader, dev_docs, rel2id, id2rel, device)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}  dev_F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), Path(args.output_dir) / "best_model.pt")
            with open(Path(args.output_dir) / "dev_predictions.json", "w") as fp:
                json.dump([list(p) for p in preds_out], fp, indent=2)
            print(f"  -> Saved best model (F1={best_f1:.4f})")


def evaluate(
    model: DocREModel,
    loader: DataLoader,
    documents: list[Document],
    rel2id: dict[str, int],
    id2rel: list[str],
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, list]:
    model.eval()
    all_preds = []
    all_gold = []
    pred_triples = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            logits = model(input_ids, attention_mask, batch["h_spans"], batch["t_spans"])
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)

            # Collect non-Na predictions as triples
            for i, (title, h_idx, t_idx) in enumerate(
                zip(batch["doc_title"], batch["h_idx"], batch["t_idx"])
            ):
                for rel_idx in range(1, len(id2rel)):  # skip Na
                    if preds[i, rel_idx] == 1:
                        pred_triples.append((title, h_idx, t_idx, id2rel[rel_idx]))

            all_preds.append(preds[:, 1:])  # exclude Na column
            all_gold.append(labels[:, 1:])

    all_preds = np.concatenate(all_preds, axis=0)
    all_gold = np.concatenate(all_gold, axis=0)
    f1 = f1_score(all_gold, all_preds, average="micro", zero_division=0)
    return f1, pred_triples


# Entry point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/re_docred")
    parser.add_argument("--output_dir", default="results/bert")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_docs", type=int, default=None, help="Limit docs for smoke testing")
    parser.add_argument("--model_name", default="bert-base-cased", help="HuggingFace model name")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
