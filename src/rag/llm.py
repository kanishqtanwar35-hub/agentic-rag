"""Generation. Gemini when a key is present, extractive fallback when not.

The fallback is not a toy: it returns the highest-scoring retrieved passage
verbatim with its citation. That means the app works offline, the tests run in
CI, and a reviewer cloning your repo sees something happen immediately instead
of hitting a "set your API key" wall.
"""

import os
import re
import time

import requests

MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")


def extractive_generate(prompt: str) -> str:
    """No-API fallback: return the first context passage with its citation.

    Crude but honest — it never invents anything, because it only ever copies.
    """
    match = re.search(r"\[1\] \(source: (.+?)\)\n(.+?)(?=\n\n\[|\n\nQuestion:)",
                      prompt, re.DOTALL)
    if not match:
        return "I don't have that in my sources."

    source, passage = match.group(1), match.group(2).strip()
    snippet = " ".join(passage.split())[:400]
    return f"{snippet} [1]\n\n(extractive mode — no GEMINI_API_KEY set; source: {source})"


def gemini_generate(prompt: str) -> str:
    # Strip whitespace and a possible UTF-8 BOM. Secrets pasted into a web form
    # or piped from a file routinely carry both, and either one fails deep in
    # http.client with a latin-1 codec error once used as a header.
    api_key = os.environ.get("GEMINI_API_KEY", "").strip().lstrip("﻿").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 600},
    }

    # Key in a header, not the query string — a `?key=...` URL leaks into
    # exception messages and from there into logs.
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    for attempt in range(3):
        r = requests.post(url, headers=headers, json=payload, timeout=45)
        if r.status_code == 429:
            time.sleep(2 ** (attempt + 2))
            continue
        r.raise_for_status()
        candidates = r.json().get("candidates") or []
        if not candidates:
            return "I don't have that in my sources."
        return candidates[0]["content"]["parts"][0]["text"]

    raise RuntimeError("rate limited after 3 attempts")


def get_generator():
    """Pick a generator based on what is available, and say which was chosen."""
    if os.environ.get("GEMINI_API_KEY"):
        return gemini_generate, "gemini"
    return extractive_generate, "extractive"
