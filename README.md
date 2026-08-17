# Agentic RAG

A retrieval system that grades its own retrieval, rewrites the query when it
fails, and **refuses to answer** when the corpus does not contain the answer.
Runs with no API key. Costs nothing.

**Status:** verified running. 15/15 tests pass, index builds, both the grounded
and the abstain paths confirmed end to end.

```
Q: Where does the oxygen from photosynthesis come from?
   grounded=True   sources=['photosynthesis.md']
   trace: retrieve -> grade -> generate

Q: What were Tesla's 2019 earnings?
   grounded=False  sources=[]
   trace: retrieve -> grade -> rewrite -> retrieve -> grade -> abstain
   answer: "I don't have that in my sources."
```

That second trace is the entire point of the project.

---

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python ingest.py                # builds artifacts/index.joblib
pytest -q                       # 15 tests, no network
uvicorn app:app --reload --port 8000
```

Open <http://localhost:8000>. The page shows the answer **and the agent's
decision trace** — every retrieval, grade, rewrite and abstain.

Optional: `set GEMINI_API_KEY=...` to switch from extractive to generative
answers. Everything works without it.

---

## Why plain RAG is not enough

The standard pipeline — embed, retrieve, stuff into a prompt, answer — has no
opinion about whether retrieval succeeded. When it returns nothing relevant,
the model answers from its own weights and sounds exactly as confident as when
it is right. That is the failure mode that makes RAG demos untrustworthy.

This adds three decisions:

| Step | What it does | Where |
|---|---|---|
| **Grade** | Drops hits below a cosine relevance floor | `agent.py: grade_hits` |
| **Rewrite** | Strips question scaffolding, retries retrieval | `agent.py: rewrite_query` |
| **Abstain** | Returns "I don't have that" without calling the model | `agent.py: answer` |

The abstain path never reaches the generator. There is a test asserting exactly
that (`test_agent_abstains_when_nothing_is_relevant`), because "the model
promised not to hallucinate" is not a guarantee — not calling it is.

---

## Design decisions worth defending

**TF-IDF instead of embeddings.** Costs nothing, needs no key, so CI and tests
actually run. It is also a real baseline: if an embedding model cannot beat
TF-IDF on your corpus, that is worth knowing. The limitation is genuine — TF-IDF
matches words, not meaning, so it misses "car" when the document says
"automobile". Swapping in `sentence-transformers` is a ~20-line change to
`store.py`; `encode()` and `search()` are the only methods the rest of the
system touches.

**Extractive fallback generator.** With no API key the system returns the
top passage verbatim with its citation. Crude, but it never invents anything,
and it means someone cloning your repo sees it work immediately.

**Ingestion separate from serving.** The API loads a prebuilt index in
milliseconds instead of re-chunking on every boot.

---

## Using your own documents

1. Drop `.md` or `.txt` files into `corpus/`
2. `python ingest.py`
3. Restart the app

Tune `max_chars` and `overlap_chars` in `chunker.py` for your content — dense
reference text wants smaller chunks, narrative prose wants larger.

---

## Deploying free

Hugging Face Spaces, Docker SDK. Free tier is 2 vCPU / 16 GB, no card required.
The `Dockerfile` builds the index at image build time and serves on 7860.

1. New Space → **Docker** SDK → Blank → Public
2. Add this front-matter to the top of `README.md` (Spaces requires it):

```yaml
---
title: Agentic RAG
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
```

3. `git push` to the Space remote

Optional: add `GEMINI_API_KEY` under Space Settings → Variables and secrets to
switch from extractive to generative answers. It works without one.

---

## Limitations, stated plainly

- TF-IDF is lexical. Synonym queries will miss. Named in the design section
  above rather than hidden.
- `rewrite_query` is a stopword filter, not an LLM rewrite. Deterministic and
  testable, but it will not rephrase a genuinely badly-formed question.
- The relevance floor (`0.08`) was tuned on a two-document corpus. Re-tune it
  for yours — measure, do not guess.
- No reranking, no hybrid search, no conversation memory. Each is a sensible
  next step.
