"""Recommendation service — LLM-generated strategic recommendations, cached locally."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.services.rag_service import (
    _client, _CHAT_MODEL, query, get_aspect_context, _build_user_message
)

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "recommendations"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SYSTEM_PROMPT = """\
You are a concise analyst writing for a railway operations team.
Write a short strategic recommendation for the given aspect using the data provided.
Keep it brief and actionable — operators are busy.

Use this exact structure (no extra sections):

## {aspect}

**Situation:** 1 sentence with the NSS and volume.

**Key issue:** 1-2 sentences on the single biggest problem from reviews.

**Actions:**
- 3-4 bullet points, each one specific and immediately actionable.

**Evidence:** 1-2 short review quotes that best illustrate the problem.

No fluff. No long explanations. Total length: under 200 words.
"""


def _cache_path(aspect: str) -> Path:
    safe = aspect.replace("/", "_").replace(" ", "_").replace("&", "and")
    return _CACHE_DIR / f"{safe}.json"


def get_recommendation(aspect: str) -> dict:
    """Return cached recommendation or generate and cache a new one."""
    path = _cache_path(aspect)

    if path.exists():
        logger.info("Recommendation cache hit: %s", aspect)
        return json.loads(path.read_text())

    logger.info("Generating recommendation for: %s", aspect)
    chunks = query(f"What are the main issues and strengths for {aspect}?", top_k=8)
    ctx = get_aspect_context(aspect, chunks)
    user_msg = _build_user_message(
        f"Write a full strategic recommendation report for the {aspect} aspect.",
        chunks,
        ctx,
    )
    system = _SYSTEM_PROMPT.replace("{aspect}", aspect)

    response = _client.chat.completions.create(
        model=_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1200,
    )

    content = response.choices[0].message.content or ""
    result = {"aspect": aspect, "content": content, "context": ctx}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info("Recommendation cached: %s", path)
    return result
