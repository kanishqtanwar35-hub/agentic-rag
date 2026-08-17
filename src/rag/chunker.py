"""Split documents into retrievable chunks.

Chunking is the most under-rated part of RAG. Chunks that are too large dilute
the signal; too small and they lose the context that makes them answerable.
Overlap exists so a fact split across a boundary still appears whole in at
least one chunk.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    position: int


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_text(
    text: str,
    source: str,
    max_chars: int = 700,
    overlap_chars: int = 120,
) -> List[Chunk]:
    """Paragraph-aware chunking with character overlap.

    Splitting on paragraph boundaries first means chunks rarely cut a sentence
    in half — better than a naive fixed-width window, and it costs three lines.
    """
    paragraphs = split_paragraphs(text)
    chunks: List[Chunk] = []
    buffer = ""
    position = 0

    def flush(buf: str) -> None:
        nonlocal position
        if buf.strip():
            chunks.append(
                Chunk(
                    id=f"{source}#{position}",
                    source=source,
                    text=buf.strip(),
                    position=position,
                )
            )
            position += 1

    for para in paragraphs:
        # A single oversized paragraph gets hard-split rather than dropped.
        if len(para) > max_chars:
            flush(buffer)
            buffer = ""
            for i in range(0, len(para), max_chars - overlap_chars):
                flush(para[i : i + max_chars])
            continue

        if len(buffer) + len(para) + 2 > max_chars:
            flush(buffer)
            buffer = buffer[-overlap_chars:] + "\n\n" + para if overlap_chars else para
        else:
            buffer = f"{buffer}\n\n{para}" if buffer else para

    flush(buffer)
    return chunks


def load_corpus(corpus_dir: Path, **kwargs) -> List[Chunk]:
    chunks: List[Chunk] = []
    files = sorted(corpus_dir.glob("**/*.md")) + sorted(corpus_dir.glob("**/*.txt"))

    if not files:
        raise FileNotFoundError(f"no .md or .txt files under {corpus_dir}")

    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_text(text, source=path.name, **kwargs))
    return chunks
