"""RAG tests that run with no API key and no network."""

import pytest

from src.rag.agent import AgenticRAG, grade_hits, rewrite_query
from src.rag.chunker import Chunk, chunk_text, load_corpus
from src.rag.store import VectorStore


@pytest.fixture
def store():
    return VectorStore().build(load_corpus_fixture())


def load_corpus_fixture():
    from pathlib import Path
    return load_corpus(Path("corpus"))


# -- chunking ----------------------------------------------------------------

def test_chunker_respects_max_chars():
    text = "\n\n".join(f"Paragraph number {i}. " * 8 for i in range(12))
    chunks = chunk_text(text, source="t.md", max_chars=400, overlap_chars=50)
    assert all(len(c.text) <= 600 for c in chunks)   # overlap adds a little
    assert len(chunks) > 1


def test_chunker_hard_splits_an_oversized_paragraph():
    text = "x" * 3000
    chunks = chunk_text(text, source="t.md", max_chars=500, overlap_chars=50)
    assert len(chunks) > 1


def test_chunk_ids_are_unique():
    chunks = load_corpus_fixture()
    assert len({c.id for c in chunks}) == len(chunks)


def test_corpus_loads_both_documents():
    sources = {c.source for c in load_corpus_fixture()}
    assert sources == {"photosynthesis.md", "water-cycle.md"}


# -- retrieval ---------------------------------------------------------------

def test_search_finds_the_right_document(store):
    hits = store.search("chlorophyll absorbs light", k=3)
    assert hits
    assert hits[0].chunk.source == "photosynthesis.md"


def test_search_finds_the_other_document(store):
    hits = store.search("evaporation from oceans", k=3)
    assert hits
    assert hits[0].chunk.source == "water-cycle.md"


def test_search_returns_nothing_for_out_of_corpus_terms(store):
    hits = store.search("quarterly revenue forecast basketball", k=4)
    assert grade_hits(hits) == []


def test_store_roundtrips_through_disk(store, tmp_path):
    path = tmp_path / "idx.joblib"
    store.save(path)
    loaded = VectorStore.load(path)
    assert len(loaded) == len(store)
    assert loaded.search("RuBisCO", k=1)[0].chunk.source == "photosynthesis.md"


# -- agent -------------------------------------------------------------------

def test_rewrite_strips_question_words():
    assert rewrite_query("What is the capital of France?") == "capital France"


def test_rewrite_never_returns_empty():
    assert rewrite_query("what is the") != ""


def test_agent_answers_from_the_corpus(store):
    agent = AgenticRAG(store, generate=lambda p: "GENERATED")
    result = agent.answer("Where does the oxygen released by photosynthesis come from?")
    assert result.grounded is True
    assert result.answer == "GENERATED"
    assert "photosynthesis.md" in result.sources


def test_agent_abstains_when_nothing_is_relevant(store):
    agent = AgenticRAG(store, generate=lambda p: "SHOULD NOT BE CALLED")
    result = agent.answer("What were Tesla's quarterly earnings in 2019?")
    assert result.grounded is False
    assert result.answer == "I don't have that in my sources."
    assert result.sources == []
    # the abstain path must never reach the generator
    assert any(s.action == "abstain" for s in result.trace)


def test_agent_trace_records_every_decision(store):
    agent = AgenticRAG(store, generate=lambda p: "ok")
    result = agent.answer("What is the Calvin cycle?")
    actions = [s.action for s in result.trace]
    assert "retrieve" in actions
    assert "grade" in actions
    assert "generate" in actions


def test_agent_rewrites_before_giving_up(store):
    agent = AgenticRAG(store, generate=lambda p: "ok", max_attempts=2)
    result = agent.answer("Tell me about xyzzy plugh frobnicate please")
    assert any(s.action == "rewrite" for s in result.trace)


def test_prompt_contains_the_abstain_instruction(store):
    captured = {}

    def spy(prompt: str) -> str:
        captured["prompt"] = prompt
        return "ok"

    AgenticRAG(store, generate=spy).answer("What is transpiration?")
    assert "I don't have that in my sources." in captured["prompt"]
    assert "ONLY the context" in captured["prompt"]
