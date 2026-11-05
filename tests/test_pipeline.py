from pathlib import Path

import numpy as np
import pytest

from rag_pipeline.cli import main
from rag_pipeline.documents import Document, chunk_document, load_documents
from rag_pipeline.embeddings import HashingEmbedder
from rag_pipeline.index import VectorIndex
from rag_pipeline.pii import redact
from rag_pipeline.pipeline import build_prompt, evaluate, ingest, retrieve

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "docs"


def test_load_documents():
    docs = load_documents(DOCS)
    assert len(docs) == 6 and all(d.text for d in docs)


def test_chunking_respects_sections_and_overlap():
    body = " ".join(f"w{i}" for i in range(250))
    doc = Document("d", "Doc", f"# Doc\n## A\n{body}\n## B\nshort section", "d.md")
    chunks = chunk_document(doc, max_words=100, overlap=10)
    a = [c for c in chunks if c.section == "A"]
    assert len(a) == 3 and a[1].text.split()[0] == "w90"
    assert any(c.section == "B" and c.text == "short section" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_bad_overlap():
    with pytest.raises(ValueError):
        chunk_document(Document("d", "t", "x", "s"), max_words=10, overlap=10)


def test_redaction():
    text, counts = redact("Mail jo@example.com or call 555-010-0199, SSN 123-45-6789")
    assert "[EMAIL]" in text and "[PHONE]" in text and "[SSN]" in text
    assert "example.com" not in text and counts["SSN"] == 1


def test_embeddings_are_normalised_and_deterministic():
    e = HashingEmbedder(dim=256)
    a, b = e.embed(["cluster cost budget"]), e.embed(["cluster cost budget"])
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)


def test_ingest_redacts_pii_before_indexing():
    index, stats = ingest(DOCS)
    assert stats.documents == 6 and stats.chunks >= 6
    assert stats.redactions.get("EMAIL") == 1 and stats.redactions.get("PHONE") == 1
    assert not any("records@example.com" in c.text for c in index.chunks)


def test_reingest_is_idempotent():
    index, _ = ingest(DOCS)
    n = len(index.chunks)
    index.add(index.chunks, index.vectors)
    assert len(index.chunks) == n


def test_retrieval_quality_meets_threshold():
    index, _ = ingest(DOCS)
    report = evaluate(index, ROOT / "data" / "eval_questions.jsonl", k=3)
    assert report["hit_rate"] >= 0.9, report
    assert report["mrr"] >= 0.75, report


def test_persistence_roundtrip(tmp_path):
    index, _ = ingest(DOCS)
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    q = "Who approves confidential table access?"
    assert retrieve(index, q)[0][0].chunk_id == retrieve(loaded, q)[0][0].chunk_id


def test_prompt_is_grounded():
    index, _ = ingest(DOCS)
    hits = retrieve(index, "When do interactive clusters shut down?")
    prompt = build_prompt("When do interactive clusters shut down?", hits)
    assert "[1]" in prompt and "only the sources" in prompt


def test_cli_end_to_end(tmp_path, capsys):
    idx = tmp_path / "idx"
    assert main(["ingest", "--docs", str(DOCS), "--index", str(idx)]) == 0
    assert main(["ask", "What is stored in the Silver layer?", "--index", str(idx)]) == 0
    assert "pipeline-standards" in capsys.readouterr().out
    assert main(["eval", "--index", str(idx), "--questions", str(ROOT / "data" / "eval_questions.jsonl"),
                 "--min-hit-rate", "0.9"]) == 0
