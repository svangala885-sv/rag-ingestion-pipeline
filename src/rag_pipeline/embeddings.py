"""Embedding backends.

`HashingEmbedder` is a dependency-free, deterministic lexical embedder so the
pipeline and its tests run offline with no API keys. `BedrockEmbedder` shows how
a managed model plugs in behind the same interface; it is imported lazily and
only used when explicitly selected.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol

import numpy as np

STOPWORDS = set(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was what "
    "when where which who will with do does can my our we you your".split()
)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def tokenize(text: str) -> list[str]:
    tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]
    # crude suffix stripping so "approvals" and "approval" share a feature
    return [re.sub(r"(ing|ed|es|s)$", "", t) if len(t) > 4 else t for t in tokens]


class HashingEmbedder:
    def __init__(self, dim: int = 1024, use_bigrams: bool = True):
        self.dim = dim
        self.use_bigrams = use_bigrams

    def _features(self, text: str) -> list[str]:
        toks = tokenize(text)
        feats = list(toks)
        if self.use_bigrams:
            feats += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat in self._features(text):
                h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
            # sublinear scaling then L2 normalise
            out[i] = np.sign(out[i]) * np.log1p(np.abs(out[i]))
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class BedrockEmbedder:  # pragma: no cover - requires AWS credentials
    """Amazon Titan text embeddings via Bedrock. Credentials come from the standard AWS chain."""

    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0", dim: int = 1024, region: str | None = None):
        import boto3  # lazy import keeps the core install light

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id, self.dim = model_id, dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            body = json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True})
            resp = self.client.invoke_model(modelId=self.model_id, body=body)
            vectors.append(json.loads(resp["body"].read())["embedding"])
        return np.asarray(vectors, dtype=np.float32)
