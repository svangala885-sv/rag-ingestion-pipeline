"""A small persistent vector index (cosine similarity) with metadata filtering.

For production volumes the same interface would sit over OpenSearch k-NN,
pgvector, Databricks Vector Search or Azure AI Search. Keeping it in numpy here
makes the retrieval logic easy to read and test.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .documents import Chunk


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        existing = {c.chunk_id for c in self.chunks}
        keep = [i for i, c in enumerate(chunks) if c.chunk_id not in existing]  # idempotent re-ingest
        self.chunks += [chunks[i] for i in keep]
        self.vectors = np.vstack([self.vectors, vectors[keep]])

    def search(self, query_vec: np.ndarray, k: int = 3, doc_id: str | None = None) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        scores = self.vectors @ query_vec.reshape(-1)
        order = np.argsort(-scores)
        hits = []
        for i in order:
            chunk = self.chunks[i]
            if doc_id and chunk.doc_id != doc_id:
                continue
            hits.append((chunk, float(scores[i])))
            if len(hits) == k:
                break
        return hits

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(asdict(c)) + "\n")

    @classmethod
    def load(cls, folder: str | Path) -> VectorIndex:
        folder = Path(folder)
        vectors = np.load(folder / "vectors.npy")
        idx = cls(vectors.shape[1])
        with open(folder / "chunks.jsonl", encoding="utf-8") as fh:
            idx.chunks = [Chunk(**json.loads(line)) for line in fh]
        idx.vectors = vectors
        return idx
