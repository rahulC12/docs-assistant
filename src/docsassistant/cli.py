"""Command-line access to the same library the web app uses.

    docs-assistant add policy.pdf handbook.docx
    docs-assistant ask "How many days of annual leave?"
    docs-assistant list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .answer import answer_question, build_llm
from .chunk import chunk_document
from .extract import ExtractionError, extract
from .index import Library

DATA_FILE = Path(os.environ.get("DATA_FILE", ".data/library.json"))


def _library() -> Library:
    return Library.load(DATA_FILE)


def cmd_add(args: argparse.Namespace) -> int:
    library = _library()
    added = 0
    for path in args.paths:
        if not path.exists():
            print(f"skipped {path}: no such file", file=sys.stderr)
            continue
        try:
            doc = extract(path.name, path.read_bytes())
        except ExtractionError as exc:
            print(f"skipped {path.name}: {exc}", file=sys.stderr)
            continue
        chunks = chunk_document(doc)
        library.add_document(doc.doc_id, doc.filename, chunks, doc.page_count)
        print(f"added {doc.filename}  {doc.page_count} pages, {len(chunks)} passages")
        added += 1

    library.save(DATA_FILE)
    return 0 if added else 1


def cmd_list(_: argparse.Namespace) -> int:
    library = _library()
    if not library.documents:
        print("Library is empty. Add a document with: docs-assistant add FILE")
        return 0
    for doc in library.documents.values():
        print(f"{doc['filename']:40} {doc['pages']:>4} pages  {doc['chunks']:>4} passages")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    library = _library()
    if not library.documents:
        print("Library is empty. Add a document first.", file=sys.stderr)
        return 1

    hits = library.search(args.question, top_k=args.top_k)
    coverage = library.coverage(args.question, hits[0].chunk) if hits else 0.0
    result = answer_question(
        args.question,
        hits,
        llm=build_llm(os.environ.get("LLM_PROVIDER", "none")),
        coverage=coverage,
        unknown_ratio=library.unknown_term_ratio(args.question),
    )

    print()
    print(result.text)
    if result.citations:
        print("\nSources")
        for c in result.citations:
            print(f"  [{c.marker}] {c.filename}, page {c.page}")
            print(f"      {c.quote[:150]}")
    print(f"\nconfidence: {result.confidence}   ({result.elapsed_ms} ms)")
    return 0 if result.grounded else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="docs-assistant",
        description="Ask questions of your documents, with page-level citations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="index one or more documents")
    add.add_argument("paths", nargs="+", type=Path)
    add.set_defaults(func=cmd_add)

    listing = sub.add_parser("list", help="show what is indexed")
    listing.set_defaults(func=cmd_list)

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
