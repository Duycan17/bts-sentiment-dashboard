"""Rule-based aspect extraction. Output guaranteed in CANONICAL_ASPECTS."""

from __future__ import annotations

import re
from functools import lru_cache

from src import config


@lru_cache(maxsize=1)
def _compiled_rules() -> list[tuple[str, re.Pattern[str]]]:
    return [(aspect, re.compile(pattern, re.IGNORECASE)) for aspect, pattern in config.ASPECT_RULES]


def extract_aspect(text: str) -> str:
    """First matching rule wins. Falls back to ASPECT_FALLBACK."""
    s = str(text)
    for aspect, pattern in _compiled_rules():
        if pattern.search(s):
            return aspect
    return config.ASPECT_FALLBACK


def extract_aspects_batch(texts: list[str]) -> list[str]:
    return [extract_aspect(t) for t in texts]
