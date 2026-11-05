"""End-to-end ingestion and retrieval, plus a retrieval-quality evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .documents import Chunk, chunk_document, load_documents
from .embeddings import Embedder, HashingEmbedder
from .index import VectorIndex
from .pii import redact

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    documents: int
    chunks: int
    redactions: dict


def ingest(folder: str | Path, embedder: Embedder | None = None, max_words: int = 120,
           overlap: int = 20) -> tuple[VectorIndex, IngestStats]:
    embedder = embedder or HashingEmbedder()
    docs = load_documents(folder)
    if not docs:
        raise ValueError(f"no .md or .txt documents found in {folder}")
    chunks: list[Chunk] = []
    redactions: dict[str, int] = {}
    for doc in docs:
        for chunk in chunk_document(doc, max_words=max_words, overlap=overlap):
            chunk.text, found = redact(chunk.text)
            for k, v in found.items():
                redactions[k] = redactions.get(k, 0) + v
            chunks.append(chunk)
    # embed title + section + text so short chunks keep their context
    vectors = embedder.embed([f"{c.title}. {c.section}. {c.text}" for c in chunks])
    index = VectorIndex(embedder.dim)
    index.add(chunks, vectors)
    stats = IngestStats(documents=len(docs), chunks=len(chunks), redactions=redactions)
    log.info("ingested %s documents into %s chunks; redactions=%s", stats.documents, stats.chunks, redactions)
    return index, stats


def retrieve(index: VectorIndex, question: str, embedder: Embedder | None = None, k: int = 3):
    embedder = embedder or HashingEmbedder()
    return index.search(embedder.embed([question])[0], k=k)


def build_prompt(question: str, hits) -> str:
    """Grounded prompt with numbered sources, for whichever LLM the caller uses."""
    context = "\n\n".join(f"[{i}] ({c.title} / {c.section})\n{c.text}" for i, (c, _) in enumerate(hits, 1))
    return (
        "Answer the question using only the sources below. Cite sources as [n]. "
        "If the sources do not contain the answer, say you do not know.\n\n"
        f"Sources:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def evaluate(index: VectorIndex, questions_path: str | Path, embedder: Embedder | None = None,
             k: int = 3) -> dict:
    """Hit rate@k and MRR over a labelled question set (question -> expected doc_id)."""
    rows = [json.loads(line) for line in Path(questions_path).read_text().splitlines() if line.strip()]
    hits_at_k, rr, misses = 0, 0.0, []
    for row in rows:
        results = retrieve(index, row["question"], embedder, k=k)
        ranks = [i for i, (c, _) in enumerate(results, 1) if c.doc_id == row["expected_doc"]]
        if ranks:
            hits_at_k += 1
            rr += 1.0 / ranks[0]
        else:
            misses.append(row["question"])
    n = len(rows)
    return {"questions": n, "k": k, "hit_rate": round(hits_at_k / n, 3), "mrr": round(rr / n, 3), "misses": misses}
