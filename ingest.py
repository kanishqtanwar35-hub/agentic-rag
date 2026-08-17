"""Build the index.

    python ingest.py

Re-run whenever the corpus changes. Keeping ingestion separate from serving is
deliberate: the API should start in milliseconds by loading a prebuilt index,
not by re-chunking every document on boot.
"""

import sys
from pathlib import Path

from src.rag.chunker import load_corpus
from src.rag.store import VectorStore

CORPUS_DIR = Path("corpus")
INDEX_PATH = Path("artifacts/index.joblib")


def main() -> int:
    print(f"reading corpus from {CORPUS_DIR}/")
    chunks = load_corpus(CORPUS_DIR)
    print(f"  {len(chunks)} chunks from "
          f"{len({c.source for c in chunks})} documents")

    store = VectorStore().build(chunks)
    store.save(INDEX_PATH)

    vocab = len(store.vectorizer.vocabulary_)
    print(f"  vocabulary: {vocab} terms")
    print(f"saved index to {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
