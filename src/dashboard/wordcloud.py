"""Word-cloud helper that returns a PIL image for in-memory rendering in Streamlit."""

from __future__ import annotations

import re
from collections import Counter
from io import BytesIO

from PIL import Image
from wordcloud import WordCloud

from src.dashboard.charts import _STOPWORDS, _tokenize  # reuse the same tokenizer + stopwords


def render_wordcloud_png(
    texts: tuple[str, ...],
    *,
    colormap: str = "viridis",
    max_words: int = 150,
    width: int = 900,
    height: int = 450,
) -> bytes:
    """Returns PNG bytes. Cached on the (texts, colormap, max_words) key."""
    tokens: list[str] = []
    for t in texts:
        tokens.extend(_tokenize(t))
    if not tokens:
        return b""
    freq = Counter(tokens)
    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        colormap=colormap,
        max_words=max_words,
        collocations=False,
    ).generate_from_frequencies(freq)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()
