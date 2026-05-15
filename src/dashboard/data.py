"""Data loading, filtering, and KPI aggregation for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import functools

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "DATA_QUALITY_CHECKLIST_filled.csv"


@dataclass
class Filters:
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
    aspects: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    bts_lines: list[str] = field(default_factory=list)
    sentiments: list[str] = field(default_factory=list)
    min_confidence: float = 0.0
    label_source: str = "ground_truth"  # or "predicted"


@functools.lru_cache(maxsize=1)
def load_dashboard_df(path: str = str(DATA_PATH)) -> pd.DataFrame:
    """Load + parse the labelled CSV. Cached across reruns."""
    df = pd.read_csv(path)
    df["created_at_date"] = pd.to_datetime(
        df["created_at_date"], errors="coerce", format="mixed"
    )
    df = df.dropna(subset=["created_at_date"]).reset_index(drop=True)
    # Drop 2026-04: these rows carry the Reddit scrape date, not the actual review date.
    # 1,831 of 1,832 rows in that month are Reddit posts scraped on 2026-04-03.
    df = df[~((df["created_at_date"].dt.year == 2026) & (df["created_at_date"].dt.month == 4))].reset_index(drop=True)
    df["year_month"] = df["created_at_date"].dt.to_period("M").astype(str)
    df["year"] = df["created_at_date"].dt.year
    df["month"] = df["created_at_date"].dt.month
    df["text_length"] = df["review_text"].astype(str).str.len()
    df["word_count"] = df["review_text"].astype(str).str.split().str.len()
    return df


def aspect_col_for(filters: Filters) -> str:
    return "aspect_pred" if filters.label_source == "predicted" else "aspect"


def sentiment_col_for(filters: Filters) -> str:
    return "sentiment_pred" if filters.label_source == "predicted" else "sentiment"


def apply_filters(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    out = df
    if filters.date_range is not None:
        start, end = filters.date_range
        out = out[(out["created_at_date"] >= start) & (out["created_at_date"] <= end)]
    if filters.aspects:
        col = aspect_col_for(filters)
        out = out[out[col].isin(filters.aspects)]
    if filters.sources:
        out = out[out["source"].isin(filters.sources)]
    if filters.bts_lines:
        out = out[out["bts_line"].isin(filters.bts_lines)]
    if filters.sentiments:
        col = sentiment_col_for(filters)
        out = out[out[col].isin(filters.sentiments)]
    if filters.min_confidence > 0:
        out = out[out["sentiment_confidence"] >= filters.min_confidence]
    return out.reset_index(drop=True)


def compute_nss(df: pd.DataFrame, sentiment_col: str = "sentiment") -> float:
    n = len(df)
    if n == 0:
        return 0.0
    pos = int((df[sentiment_col] == "Positive").sum())
    neg = int((df[sentiment_col] == "Negative").sum())
    return float((pos - neg) / n * 100.0)


def compute_kpis(df: pd.DataFrame, sentiment_col: str = "sentiment") -> dict:
    n = max(len(df), 1)
    counts = df[sentiment_col].value_counts()
    pos = int(counts.get("Positive", 0))
    neg = int(counts.get("Negative", 0))
    neu = int(counts.get("Neutral", 0))
    return {
        "rows": int(len(df)),
        "nss": compute_nss(df, sentiment_col),
        "pct_positive": pos / n * 100,
        "pct_neutral": neu / n * 100,
        "pct_negative": neg / n * 100,
        "n_aspects": int(df["aspect"].nunique()),
        "date_min": df["created_at_date"].min(),
        "date_max": df["created_at_date"].max(),
        "n_sources": int(df["source"].nunique()),
        "avg_confidence": float(df["sentiment_confidence"].mean()) if len(df) else 0.0,
    }


def aspect_nss_table(df: pd.DataFrame, *, aspect_col: str = "aspect", sentiment_col: str = "sentiment") -> pd.DataFrame:
    grp = df.groupby(aspect_col).agg(
        rows=(sentiment_col, "size"),
        pos=(sentiment_col, lambda s: (s == "Positive").sum()),
        neg=(sentiment_col, lambda s: (s == "Negative").sum()),
        neu=(sentiment_col, lambda s: (s == "Neutral").sum()),
    )
    grp["nss"] = (grp["pos"] - grp["neg"]) / grp["rows"].clip(lower=1) * 100
    grp["pct_negative"] = grp["neg"] / grp["rows"].clip(lower=1) * 100
    return grp.reset_index().rename(columns={aspect_col: "aspect"}).sort_values("nss")
