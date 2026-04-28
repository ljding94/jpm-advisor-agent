"""CLI: ingest data/knowledge_base/*.md into the persisted Chroma store."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.tools.knowledge_store import KnowledgeStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest knowledge base markdown into Chroma.")
    parser.add_argument(
        "--source", default="data/knowledge_base", help="Directory of markdown files."
    )
    parser.add_argument(
        "--persist", default=os.getenv("CHROMA_PATH", "data/chroma"),
        help="ChromaDB persistence directory."
    )
    parser.add_argument("--reset", action="store_true", help="Drop existing collection first.")
    args = parser.parse_args()

    store = KnowledgeStore(persist_path=args.persist)
    if args.reset:
        store.reset()
    n = store.ingest_directory(Path(args.source))
    print(f"Ingested {n} chunks from {args.source!r} → {args.persist!r}")
    print(f"Collection size: {store.count()} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
