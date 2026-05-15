"""RAG service — local ChromaDB + sentence-transformers embeddings + OpenAI-compatible LLM."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Generator

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.dashboard.data import load_dashboard_df

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

# ── LLM config ────────────────────────────────────────────────────────────────
_API_KEY = os.environ["LLM_API_KEY"]
_BASE_URL = os.environ["LLM_BASE_URL"]
_CHAT_MODEL = os.environ.get("LLM_CHAT_MODEL", "gpt-5.4")
_CHROMA_PATH = str(Path(__file__).resolve().parents[2] / "artifacts" / "chroma")
_COLLECTION = "bts_reviews"
_SAMPLES_PER_CLASS = 6
_EMBED_MODEL = "all-MiniLM-L6-v2"  # via chromadb DefaultEmbeddingFunction (onnxruntime)

_client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)

# ── Lazy singletons ───────────────────────────────────────────────────────────
_embedder = None
_collection = None

KNOWN_ASPECTS = [
    "Accessibility",
    "Cleanliness",
    "Crowding & Comfort",
    "Fare & Payment System",
    "Information & Navigation",
    "Overall Experience & Convenience",
    "Route Coverage & Connectivity",
    "Safety & Security",
    "Staff & Service",
    "Train Frequency & Waiting Time",
]

_ACTIONS: dict[str, str] = {
    "Accessibility": "Install lifts at 5 worst-rated stations; publish 90-day remediation roadmap.",
    "Information & Navigation": "Deploy bilingual real-time displays; redesign in-app station maps.",
    "Safety & Security": "Increase off-peak patrols; publish monthly incident transparency report.",
    "Staff & Service": "Launch service-excellence training; add English-language complaint hotline.",
    "Crowding & Comfort": "Surface live carriage-load in app; add express services on peak corridors.",
    "Cleanliness": "Double cleaning frequency at top-3 negative stations; introduce QA photo audit.",
    "Train Frequency & Waiting Time": "Maintain dispatch interval; publicise on-time stats as brand asset.",
    "Fare & Payment System": "Promote Rabbit Card bundles; expand contactless payment at all gates.",
    "Route Coverage & Connectivity": "Lead marketing with network reach; improve MRT interchange signage.",
    "Overall Experience & Convenience": "Protect NPS with strict SLAs; use positive reviews in campaigns.",
}


# ── Embedder (ChromaDB DefaultEmbeddingFunction — onnxruntime, no PyTorch) ────

def _get_embedder():
    global _embedder
    if _embedder is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _embedder = DefaultEmbeddingFunction()
    return _embedder


def _embed(texts: list[str]) -> list[list[float]]:
    return list(_get_embedder()(texts))


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    import chromadb
    chroma = chromadb.PersistentClient(path=_CHROMA_PATH)
    _collection = chroma.get_or_create_collection(
        name=_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


# ── Vector store build ────────────────────────────────────────────────────────

def build_vector_store(force: bool = False) -> None:
    """Seed ChromaDB with 6 neg + 6 pos reviews per aspect. Idempotent."""
    col = _get_collection()
    if col.count() > 0 and not force:
        logger.info("RAG: vector store already built (%d docs), skipping.", col.count())
        return

    logger.info("RAG: building vector store…")
    df = load_dashboard_df()

    docs, ids, metas = [], [], []
    for aspect in df["aspect"].dropna().unique():
        sub = df[df["aspect"] == aspect]
        neg = sub[sub["sentiment"] == "Negative"].head(_SAMPLES_PER_CLASS)
        pos = sub[sub["sentiment"] == "Positive"].head(_SAMPLES_PER_CLASS)
        for _, row in pd.concat([neg, pos]).iterrows():
            text = str(row["review_text"])[:1000]
            doc_id = f"{aspect}_{row.name}"
            docs.append(text)
            ids.append(str(doc_id))
            metas.append({
                "aspect": str(aspect),
                "sentiment": str(row["sentiment"]),
                "source": str(row.get("source", "")),
                "date": str(row.get("created_at_date", ""))[:10],
                "rating": int(row["review_rating"]) if pd.notna(row.get("review_rating")) else 0,
            })

    embeddings = _embed(docs)
    col.upsert(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
    logger.info("RAG: upserted %d documents.", len(docs))


# ── Query ─────────────────────────────────────────────────────────────────────

def query(question: str, top_k: int = 8) -> list[dict]:
    col = _get_collection()
    q_emb = _embed([question])[0]
    results = col.query(query_embeddings=[q_emb], n_results=min(top_k, col.count()))
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        chunks.append({
            "text": doc,
            "aspect": meta.get("aspect", ""),
            "sentiment": meta.get("sentiment", ""),
            "source": meta.get("source", ""),
            "date": meta.get("date", ""),
            "rating": meta.get("rating", 0),
        })
    return chunks


# ── Aspect detection ──────────────────────────────────────────────────────────

def detect_aspect(question: str) -> str | None:
    q_lower = question.lower()
    for asp in KNOWN_ASPECTS:
        words = [w.lower() for w in asp.replace("&", "").split() if len(w) > 3]
        if any(w in q_lower for w in words):
            return asp
    return None


# ── Aspect business context ───────────────────────────────────────────────────

def get_aspect_context(aspect: str | None, chunks: list[dict]) -> dict:
    if aspect is None and chunks:
        aspect = Counter(c["aspect"] for c in chunks).most_common(1)[0][0]
    if aspect is None:
        return {}

    df = load_dashboard_df()
    sub = df[df["aspect"] == aspect]
    if sub.empty:
        return {"aspect": aspect, "action": _ACTIONS.get(aspect, "")}

    n = len(sub)
    pos = (sub["sentiment"] == "Positive").sum()
    neg = (sub["sentiment"] == "Negative").sum()
    neu = (sub["sentiment"] == "Neutral").sum()
    nss = round((pos - neg) / n * 100, 1) if n else 0.0

    # YoY NSS
    yoy = (
        sub.groupby("year")
        .apply(lambda g: round((( g["sentiment"] == "Positive").sum() - (g["sentiment"] == "Negative").sum()) / len(g) * 100, 1))
        .to_dict()
    )

    # Monthly trend (last 12 months)
    monthly = (
        sub.groupby("year_month")
        .apply(lambda g: round(((g["sentiment"] == "Positive").sum() - (g["sentiment"] == "Negative").sum()) / len(g) * 100, 1))
        .tail(12)
        .to_dict()
    )

    # Recent vs prior NSS
    months = sorted(sub["year_month"].unique())
    mid = len(months) // 2
    recent = sub[sub["year_month"].isin(months[mid:])]
    prior = sub[sub["year_month"].isin(months[:mid])]
    def _nss(g): return round(((g["sentiment"] == "Positive").sum() - (g["sentiment"] == "Negative").sum()) / max(len(g), 1) * 100, 1)
    recent_nss = _nss(recent)
    prior_nss = _nss(prior)
    trend_delta = round(recent_nss - prior_nss, 1)
    trend_dir = "improving" if trend_delta > 5 else "declining" if trend_delta < -5 else "stable"

    # Source breakdown
    src = (
        sub.groupby("source")
        .apply(lambda g: {"rows": len(g), "nss": _nss(g)})
        .to_dict()
    )

    # BTS line breakdown
    line_col = sub["bts_line"].dropna()
    line_breakdown = (
        sub[sub["bts_line"].notna()].groupby("bts_line")
        .apply(lambda g: {"rows": len(g), "nss": _nss(g)})
        .to_dict()
    ) if not line_col.empty else {}

    # Rating distribution
    rating_dist = sub["review_rating"].value_counts().sort_index().to_dict()
    avg_rating = round(sub["review_rating"].mean(), 2)

    # Top reviewer hometowns
    top_hometowns = (
        sub["reviewer_hometown"].dropna()
        .value_counts()
        .head(5)
        .to_dict()
    )

    # Top bigrams from negative reviews (reuse tokenizer from business_service)
    import re
    _SW = {"the","a","an","and","or","but","in","on","at","to","for","of","with","is","it","i","you","we","they","this","that","was","are","be","have","has","had","do","did","not","from","by","as","so","if","my","me","he","she","his","her","our","your","their","its","will","would","can","could","should","there","here","what","which","who","how","when","where","why","all","any","some","no","more","also","just","up","out","about","than","then","been","were","am","into","very","get","got","one","two","like","go","use","used","amp","re","s","t","m","bts","mrt","skytrain","bangkok","thailand","thai"}
    def _bigrams(texts, n=5):
        c: Counter = Counter()
        for t in texts:
            toks = [w for w in re.findall(r"\b[a-z]{3,}\b", t.lower()) if w not in _SW]
            for i in range(len(toks)-1): c[f"{toks[i]} {toks[i+1]}"] += 1
        return [bg for bg, _ in c.most_common(n)]

    neg_texts = sub[sub["sentiment"] == "Negative"]["review_text"].astype(str).head(300).tolist()
    pos_texts = sub[sub["sentiment"] == "Positive"]["review_text"].astype(str).head(300).tolist()

    return {
        "aspect": aspect,
        "nss": nss,
        "rows": n,
        "pct_positive": round(pos / n * 100, 1) if n else 0,
        "pct_negative": round(neg / n * 100, 1) if n else 0,
        "pct_neutral": round(neu / n * 100, 1) if n else 0,
        "trend_direction": trend_dir,
        "trend_delta_pp": trend_delta,
        "recent_nss": recent_nss,
        "prior_nss": prior_nss,
        "yoy_nss": yoy,
        "monthly_nss_last12": monthly,
        "source_breakdown": src,
        "bts_line_breakdown": line_breakdown,
        "avg_rating": avg_rating,
        "rating_distribution": rating_dist,
        "top_reviewer_hometowns": top_hometowns,
        "top_negative_bigrams": _bigrams(neg_texts),
        "top_positive_bigrams": _bigrams(pos_texts),
        "action": _ACTIONS.get(aspect, f"Run a focused improvement sprint on {aspect}."),
    }


# ── Streaming chat ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior business intelligence analyst for Bangkok BTS Skytrain.
Answer the user's question using the provided review excerpts and statistics.
Structure every answer as:
1. **Direct answer** (2-3 sentences)
2. **Key data points** (bullet list with numbers from the stats)
3. **Recommended actions** (from the business data)
4. **Data story** (what the trend means for the business)

Be concise, data-driven, and actionable. Do not invent data not present in the context.
"""


def _build_user_message(question: str, chunks: list[dict], ctx: dict) -> str:
    chunk_text = "\n\n".join(
        f"[{c['aspect']} | {c['sentiment']} | {c['date']} | {c['source']}]\n{c['text']}"
        for c in chunks
    )
    stats = ""
    if ctx:
        yoy = ", ".join(f"{y}: {v:+.1f}%" for y, v in sorted(ctx.get("yoy_nss", {}).items()))
        monthly = ", ".join(f"{m}: {v:+.1f}%" for m, v in list(ctx.get("monthly_nss_last12", {}).items())[-6:])
        src = "; ".join(f"{s}: NSS {v['nss']:+.1f}% ({v['rows']} reviews)" for s, v in ctx.get("source_breakdown", {}).items())
        lines = "; ".join(f"{l}: NSS {v['nss']:+.1f}% ({v['rows']} reviews)" for l, v in ctx.get("bts_line_breakdown", {}).items())
        rating = ", ".join(f"★{k}: {v}" for k, v in sorted(ctx.get("rating_distribution", {}).items()))
        hometowns = ", ".join(f"{h} ({c})" for h, c in ctx.get("top_reviewer_hometowns", {}).items())
        neg_bigrams = ", ".join(ctx.get("top_negative_bigrams", []))
        pos_bigrams = ", ".join(ctx.get("top_positive_bigrams", []))

        stats = f"""
Live statistics for **{ctx.get('aspect', '')}**:
- NSS: {ctx.get('nss', 'N/A')}% | Volume: {ctx.get('rows', 'N/A')} reviews
- Positive: {ctx.get('pct_positive', 'N/A')}% | Neutral: {ctx.get('pct_neutral', 'N/A')}% | Negative: {ctx.get('pct_negative', 'N/A')}%
- Trend: {ctx.get('trend_direction', 'N/A')} ({ctx.get('trend_delta_pp', 0):+.1f}pp) | Recent NSS: {ctx.get('recent_nss', 'N/A')}% | Prior NSS: {ctx.get('prior_nss', 'N/A')}%
- Year-over-year NSS: {yoy}
- Monthly NSS (last 6 months): {monthly}
- By source: {src}
- By BTS line: {lines if lines else 'N/A'}
- Avg rating: {ctx.get('avg_rating', 'N/A')} | Rating dist: {rating}
- Top reviewer hometowns: {hometowns if hometowns else 'N/A'}
- Top negative phrases: {neg_bigrams}
- Top positive phrases: {pos_bigrams}
- Recommended action: {ctx.get('action', '')}
"""
    return f"Question: {question}\n\nRetrieved reviews:\n{chunk_text}\n{stats}"


def stream_chat(message: str, history: list[dict]) -> Generator[str, None, None]:
    """Yield SSE-formatted chunks for the chat response."""
    chunks = query(message)
    aspect = detect_aspect(message)
    ctx = get_aspect_context(aspect, chunks)

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": _build_user_message(message, chunks, ctx)})

    # Yield sources first
    sources_payload = json.dumps([
        {"text": c["text"][:200], "aspect": c["aspect"], "sentiment": c["sentiment"],
         "date": c["date"], "source": c["source"]}
        for c in chunks
    ])
    yield f"event: sources\ndata: {sources_payload}\n\n"

    # Stream LLM tokens
    stream = _client.chat.completions.create(
        model=_CHAT_MODEL,
        messages=messages,
        stream=True,
        temperature=0.3,
        max_tokens=800,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield f"data: {json.dumps(delta)}\n\n"

    yield "data: [DONE]\n\n"
