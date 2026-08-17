"""FastAPI service with a minimal chat page.

The page shows the agent's TRACE alongside the answer — every retrieval,
grade, rewrite and abstain decision. That transparency is the demo. Anyone can
build a box that returns text; showing why it returned that text is the part
that reads as engineering.

Run:  uvicorn app:app --reload --port 8000
"""

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.rag.agent import AgenticRAG
from src.rag.llm import get_generator
from src.rag.store import VectorStore

INDEX_PATH = Path("artifacts/index.joblib")

app = FastAPI(title="Agentic RAG", version="1.0.0")

_agent = None
_mode = "unknown"


def get_agent() -> AgenticRAG:
    global _agent, _mode
    if _agent is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"No index at {INDEX_PATH}. Run `python ingest.py` first."
            )
        generate, _mode = get_generator()
        _agent = AgenticRAG(VectorStore.load(INDEX_PATH), generate)
    return _agent


class AskRequest(BaseModel):
    question: str


class StepOut(BaseModel):
    action: str
    detail: str
    hits: List[str] = []


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    grounded: bool
    trace: List[StepOut]


@app.get("/health")
def health() -> dict:
    try:
        agent = get_agent()
        return {
            "status": "ok",
            "chunks": len(agent.store),
            "generator": _mode,
        }
    except Exception as e:
        return {"status": "degraded", "detail": str(e)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        result = get_agent().answer(question)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=result.sources,
        grounded=result.grounded,
        trace=[StepOut(action=s.action, detail=s.detail, hits=s.hits)
               for s in result.trace],
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    try:
        agent = get_agent()
        status = f"{len(agent.store)} chunks indexed &middot; generator: {_mode}"
    except Exception as e:
        status = f"not ready — {e}"

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agentic RAG</title><style>
 :root {{ --bg:#fbfbfa;--fg:#16211f;--muted:#6b7a76;--line:#e2e6e3;--card:#fff;--ok:#2c6b4b; }}
 @media (prefers-color-scheme:dark) {{ :root {{ --bg:#0e1615;--fg:#e7ede9;--muted:#8fa39d;
   --line:#25332f;--card:#141f1d;--ok:#63c08d; }} }}
 body {{ background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif;
   margin:0;padding:2.5rem 1rem 4rem; }}
 .wrap {{ max-width:760px;margin:0 auto; }}
 h1 {{ font-size:1.5rem;margin:0 0 .2rem; }}
 .sub {{ color:var(--muted);font-size:.85rem;margin:0 0 1.5rem; }}
 form {{ display:flex;gap:.5rem;margin-bottom:1.5rem; }}
 input {{ flex:1;padding:.6rem .7rem;border:1px solid var(--line);
   background:var(--card);color:var(--fg);font-size:1rem; }}
 button {{ padding:.6rem 1.1rem;border:1px solid var(--fg);background:var(--fg);
   color:var(--bg);cursor:pointer;font-size:.95rem; }}
 .card {{ background:var(--card);border:1px solid var(--line);padding:1rem 1.1rem;
   margin-bottom:1rem; }}
 .lbl {{ font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
   color:var(--muted);margin-bottom:.4rem; }}
 .src {{ color:var(--ok);font-size:.82rem;margin-top:.6rem; }}
 .trace {{ font:.78rem ui-monospace,Menlo,Consolas,monospace;color:var(--muted); }}
 .trace div {{ padding:.15rem 0;border-bottom:1px solid var(--line); }}
 .act {{ color:var(--fg);font-weight:600; }}
</style></head><body><div class="wrap">
<h1>Agentic RAG</h1>
<p class="sub">{status}</p>
<form onsubmit="ask(event)">
  <input id="q" placeholder="Where does the oxygen from photosynthesis come from?" autofocus>
  <button type="submit">Ask</button>
</form>
<div id="out"></div>
<script>
async function ask(e) {{
  e.preventDefault();
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const out = document.getElementById('out');
  out.innerHTML = '<div class="card">thinking…</div>';
  try {{
    const r = await fetch('/ask', {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{question:q}})
    }});
    const d = await r.json();
    if (!r.ok) {{ out.innerHTML = '<div class="card">'+(d.detail||'error')+'</div>'; return; }}
    const trace = d.trace.map(s =>
      '<div><span class="act">'+s.action+'</span> — '+s.detail+'</div>').join('');
    out.innerHTML =
      '<div class="card"><div class="lbl">Answer</div>'+
      d.answer.replace(/\\n/g,'<br>')+
      (d.sources.length ? '<div class="src">sources: '+d.sources.join(', ')+'</div>' : '')+
      '</div><div class="card"><div class="lbl">Agent trace</div>'+
      '<div class="trace">'+trace+'</div></div>';
  }} catch (err) {{
    out.innerHTML = '<div class="card">request failed: '+err+'</div>';
  }}
}}
</script></div></body></html>"""
