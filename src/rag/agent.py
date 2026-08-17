"""The agentic part.

Plain RAG is: embed query -> retrieve -> stuff into prompt -> answer. It fails
silently when retrieval returns nothing relevant, because the model answers
from its own weights and sounds just as confident.

This adds three decisions the system makes for itself:

  1. GRADE   — are the retrieved chunks actually relevant to the question?
  2. REWRITE — if not, reformulate the query and retrieve again.
  3. ABSTAIN — if it still fails, say "I don't know" instead of inventing.

Step 3 is the one that matters. A RAG system that cannot say "not in the
corpus" is not a retrieval system, it is a hallucination engine with citations.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.rag.store import Hit, VectorStore

# Below this cosine score, a hit is treated as noise rather than evidence.
RELEVANCE_FLOOR = 0.08
# How many chunks must clear the floor for retrieval to count as successful.
MIN_GOOD_HITS = 1


@dataclass
class Step:
    """One decision the agent made. Surfacing these is most of the value."""

    action: str
    detail: str
    hits: List[str] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[str]
    trace: List[Step]
    grounded: bool


def grade_hits(hits: List[Hit], floor: float = RELEVANCE_FLOOR) -> List[Hit]:
    return [h for h in hits if h.score >= floor]


def rewrite_query(question: str) -> str:
    """Lexical rewrite: strip question scaffolding, keep the content words.

    TF-IDF matches words, so "What is the capital of France?" retrieves worse
    than "capital France" — the question words are noise that dilutes the
    vector. A real system would use an LLM here; this version is deterministic,
    free, and testable, which matters more for a first build.
    """
    stopwords = {
        "what", "when", "where", "who", "why", "how", "is", "are", "was",
        "were", "the", "a", "an", "of", "in", "on", "to", "for", "do", "does",
        "did", "can", "could", "would", "should", "tell", "me", "about",
        "explain", "describe", "please",
    }
    words = [w.strip("?.,!") for w in question.split()]
    kept = [w for w in words if w.lower() not in stopwords]
    return " ".join(kept) if kept else question


def build_prompt(question: str, hits: List[Hit]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] (source: {h.chunk.source})\n{h.chunk.text}"
        for i, h in enumerate(hits)
    )
    return f"""Answer the question using ONLY the context below.

If the context does not contain the answer, reply exactly:
"I don't have that in my sources."

Cite the numbered sources you used, like [1] or [2].

Context:
{context}

Question: {question}

Answer:"""


class AgenticRAG:
    def __init__(
        self,
        store: VectorStore,
        generate: Callable[[str], str],
        k: int = 4,
        max_attempts: int = 2,
    ):
        self.store = store
        self.generate = generate
        self.k = k
        self.max_attempts = max_attempts

    def answer(self, question: str) -> Answer:
        trace: List[Step] = []
        query = question
        good: List[Hit] = []

        for attempt in range(1, self.max_attempts + 1):
            hits = self.store.search(query, k=self.k)
            trace.append(
                Step(
                    action="retrieve",
                    detail=f"attempt {attempt}, query={query!r}, {len(hits)} raw hits",
                    hits=[f"{h.chunk.id} ({h.score:.3f})" for h in hits],
                )
            )

            good = grade_hits(hits)
            trace.append(
                Step(
                    action="grade",
                    detail=(
                        f"{len(good)}/{len(hits)} hits cleared the "
                        f"{RELEVANCE_FLOOR} relevance floor"
                    ),
                )
            )

            if len(good) >= MIN_GOOD_HITS:
                break

            if attempt < self.max_attempts:
                query = rewrite_query(question)
                trace.append(
                    Step(action="rewrite", detail=f"retrying with {query!r}")
                )

        if not good:
            trace.append(
                Step(action="abstain", detail="no chunk cleared the floor after rewriting")
            )
            return Answer(
                question=question,
                answer="I don't have that in my sources.",
                sources=[],
                trace=trace,
                grounded=False,
            )

        prompt = build_prompt(question, good)
        text = self.generate(prompt)
        trace.append(Step(action="generate", detail=f"{len(prompt)} chars of context"))

        return Answer(
            question=question,
            answer=text,
            sources=sorted({h.chunk.source for h in good}),
            trace=trace,
            grounded=True,
        )
