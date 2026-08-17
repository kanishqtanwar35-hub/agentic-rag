# Code walkthrough — 04-agentic-rag

```
corpus/*.md
    |
    v  chunker.load_corpus()      paragraph-aware, overlapping
 [Chunk]
    |
    v  store.VectorStore.build()  TF-IDF, 1-2 grams
 artifacts/index.joblib
    |
    v  agent.AgenticRAG.answer(question)
         |
         +-- retrieve   store.search(query, k=4)
         +-- grade      drop hits below RELEVANCE_FLOOR
         |     |
         |     +-- enough good hits? --> build_prompt --> generate --> Answer
         |     |
         |     +-- not enough, attempt 1? --> rewrite_query --> retry
         |     |
         |     +-- still nothing --> ABSTAIN (generator never called)
         v
    Answer(answer, sources, trace, grounded)
```

---

## `chunker.py`

**Paragraph-first, then pack.** Naive fixed-width chunking cuts sentences in
half and the fragment retrieves badly. Splitting on blank lines first and
packing paragraphs up to `max_chars` costs three extra lines and produces
chunks that read as coherent units.

**Overlap** (`overlap_chars=120`) exists for facts that straddle a boundary.
Without it, a sentence split across two chunks appears whole in neither.

**Oversized paragraphs get hard-split** rather than dropped. Worth noticing:
the naive version of this function silently loses any paragraph longer than
`max_chars`, and you would not find out until a query returned nothing.
`test_chunker_hard_splits_an_oversized_paragraph` pins it.

## `store.py`

`TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, stop_words="english")`.

- **Bigrams** so "water cycle" is a term, not two independent words.
- **`sublinear_tf`** applies log scaling to term frequency, so a chunk that
  says "photosynthesis" nine times does not swamp one that says it twice in a
  more relevant context.
- **`scores[i] > 0` filter** in `search` — a zero-similarity result is not a
  weak result, it is *no* result, and returning it as a "hit" is what feeds
  garbage into the prompt.

The store exposes exactly two methods the rest of the system uses: `build` and
`search`. That is the seam. Replacing TF-IDF with `sentence-transformers` or
Qdrant means rewriting this file and nothing else.

## `agent.py` — the part that matters

### `grade_hits`

One line: keep hits at or above `RELEVANCE_FLOOR = 0.08`. That constant was
tuned on a two-document corpus and **you must re-tune it on yours** — the
README says so. Too high and you abstain on answerable questions; too low and
noise reaches the prompt.

This is the cheap version of "retrieval grading". The expensive version asks an
LLM "is this chunk relevant to this question?" per chunk. Start cheap: it is
free, deterministic, and testable, and it catches the majority of failures.

### `rewrite_query`

Strips question scaffolding — "What is the capital of France?" becomes
"capital France". For a lexical retriever the question words are pure noise
diluting the query vector.

Deliberately not an LLM call. Deterministic means `test_rewrite_strips_question_words`
can assert an exact string, and the rewrite costs nothing. Upgrade path is
obvious once you have evals (project 03) to prove the upgrade helped.

Note the guard: `return " ".join(kept) if kept else question`. A query made
entirely of stopwords would otherwise rewrite to the empty string and retrieve
nothing. `test_rewrite_never_returns_empty` exists for that.

### `answer` — the loop

```python
for attempt in range(1, self.max_attempts + 1):
    hits = self.store.search(query, k=self.k)     # retrieve
    good = grade_hits(hits)                        # grade
    if len(good) >= MIN_GOOD_HITS: break           # done
    if attempt < self.max_attempts:
        query = rewrite_query(question)            # rewrite, retry
if not good:
    return Answer(..., grounded=False)             # abstain
```

**The abstain branch returns before `self.generate` is called.** Not "the
prompt tells the model to say I don't know" — the model is never invoked. A
prompt instruction is a request; not making the call is a guarantee. That
distinction is the single best thing to say about this project in an interview.

`test_agent_abstains_when_nothing_is_relevant` passes a generator that returns
`"SHOULD NOT BE CALLED"` and asserts it never appears. That is how you test a
negative.

### The trace

Every decision appends a `Step`. The UI renders them. This is most of the
demo's value: anyone can build a box that returns text, but showing *why* it
returned that text — and why it sometimes returns nothing — is what reads as
engineering rather than a wrapper.

## `llm.py`

Two generators behind one `get_generator()` that picks based on whether
`GEMINI_API_KEY` exists, and reports which it chose via `/health`.

The extractive fallback returns the top passage verbatim. It cannot hallucinate
because it only copies. That property is why the tests and CI work with no
secrets at all.

## `app.py`

`get_agent()` is lazy and cached in a module global, so the index loads once per
process. `/health` reports chunk count and generator mode — if someone says
"your demo is broken", that endpoint tells you in one request whether the index
is missing or the key is unset.

---

## What to build next, in order

1. **Swap TF-IDF for embeddings** and measure the difference using project 03's
   eval harness. Do not guess — measure. That comparison, with numbers, is a
   better portfolio artifact than either system alone.
2. **Add reranking**: retrieve 20, rerank to 4 with a cross-encoder.
3. **Add conversation memory** — resolve "what about the other one?" against
   the previous turn.
4. **Log every abstain.** The questions your corpus cannot answer are a ranked
   list of what to add to it.
