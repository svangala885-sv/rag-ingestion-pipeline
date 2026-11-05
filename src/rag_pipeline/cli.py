"""CLI: ingest a folder, ask a question, or run the retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import logging

from .index import VectorIndex
from .pipeline import build_prompt, evaluate, ingest, retrieve


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest")
    a.add_argument("--docs", default="data/docs")
    a.add_argument("--index", default="build/index")
    q = sub.add_parser("ask")
    q.add_argument("question")
    q.add_argument("--index", default="build/index")
    q.add_argument("-k", type=int, default=3)
    q.add_argument("--show-prompt", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--index", default="build/index")
    e.add_argument("--questions", default="data/eval_questions.jsonl")
    e.add_argument("-k", type=int, default=3)
    e.add_argument("--min-hit-rate", type=float, default=0.0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "ingest":
        index, stats = ingest(args.docs)
        index.save(args.index)
        print(json.dumps(stats.__dict__))
        return 0
    index = VectorIndex.load(args.index)
    if args.cmd == "ask":
        hits = retrieve(index, args.question, k=args.k)
        for chunk, score in hits:
            print(f"{score:.3f}  {chunk.doc_id} / {chunk.section}: {chunk.text[:110]}...")
        if args.show_prompt:
            print("\n" + build_prompt(args.question, hits))
        return 0
    report = evaluate(index, args.questions, k=args.k)
    print(json.dumps(report, indent=2))
    return 0 if report["hit_rate"] >= args.min_hit_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
