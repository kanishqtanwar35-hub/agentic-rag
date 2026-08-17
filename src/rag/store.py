"""A TF-IDF vector store.

Why not embeddings? Three reasons, and they are all defensible in an interview:

  1. It costs nothing and needs no API key, so the tests and CI actually run.
  2. It is a genuine baseline. If your fancy embedding model cannot beat
     TF-IDF on your corpus, you have learned something important.
  3. Swapping it out is a 20-line change — `encode()` and `search()` are the
     only methods the rest of the system knows about.

TF-IDF is lexical: it matches words, not meaning. It will miss "car" when the
document says "automobile". That limitation is stated in the README rather than
hidden, and it is exactly what the query-rewriting step below compensates for.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.chunker import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),      # bigrams catch "water cycle" as a unit
            sublinear_tf=True,       # dampens repeated-term dominance
            min_df=1,
        )
        self.chunks: List[Chunk] = []
        self.matrix = None

    def build(self, chunks: List[Chunk]) -> "VectorStore":
        if not chunks:
            raise ValueError("cannot build a store from zero chunks")
        self.chunks = chunks
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])
        return self

    def search(self, query: str, k: int = 4) -> List[Hit]:
        if self.matrix is None:
            raise RuntimeError("store is empty — call build() or load() first")

        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]

        # argsort ascending, take the tail, reverse — cheaper than a full sort
        top = np.argsort(scores)[-k:][::-1]
        return [
            Hit(chunk=self.chunks[i], score=float(scores[i]))
            for i in top
            if scores[i] > 0          # a zero score is not a result
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "chunks": self.chunks, "matrix": self.matrix},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        data = joblib.load(path)
        store = cls()
        store.vectorizer = data["vectorizer"]
        store.chunks = data["chunks"]
        store.matrix = data["matrix"]
        return store

    def __len__(self) -> int:
        return len(self.chunks)
