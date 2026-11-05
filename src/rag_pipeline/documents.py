"""Load source documents and split them into overlapping, heading-aware chunks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".md", ".txt"}


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_documents(folder: str | Path) -> list[Document]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"document folder not found: {folder}")
    docs = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        first = text.splitlines()[0].lstrip("# ").strip()
        docs.append(Document(doc_id=path.stem, title=first or path.stem, text=text, source=str(path.name)))
    return docs


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs so chunks never straddle two topics."""
    parts: list[tuple[str, str]] = []
    heading, buf = "", []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if any(b.strip() for b in buf):
                parts.append((heading, "\n".join(buf).strip()))
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if any(b.strip() for b in buf):
        parts.append((heading, "\n".join(buf).strip()))
    return parts


def chunk_document(doc: Document, max_words: int = 120, overlap: int = 20) -> list[Chunk]:
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    chunks = []
    for section, body in _sections(doc.text):
        words = body.split()
        step = max_words - overlap
        for start in range(0, max(len(words), 1), step):
            window = words[start : start + max_words]
            if not window:
                break
            text = " ".join(window)
            digest = hashlib.sha1(f"{doc.doc_id}|{section}|{start}|{text}".encode()).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-{digest}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=section,
                    text=text,
                    metadata={"source": doc.source, "word_start": start},
                )
            )
            if start + max_words >= len(words):
                break
    return chunks
