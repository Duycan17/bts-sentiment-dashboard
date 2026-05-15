"""Business insights service — agency-grade analytics for the C-suite."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.dashboard.data import (
    Filters,
    apply_filters,
    aspect_nss_table,
    load_dashboard_df,
    sentiment_col_for,
)
from backend.models import (
    AspectDeepDive,
    BusinessResponse,
    ExecutiveSummary,
    SourceBreakdown,
    TrendDirection,
)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "is", "it", "i", "you", "we", "they", "this", "that", "was", "are",
    "be", "have", "has", "had", "do", "did", "not", "from", "by", "as", "so",
    "if", "my", "me", "he", "she", "his", "her", "our", "your", "their", "its",
    "will", "would", "can", "could", "should", "there", "here", "what", "which",
    "who", "how", "when", "where", "why", "all", "any", "some", "no", "more",
    "also", "just", "up", "out", "about", "than", "then", "been", "were", "am",
    "into", "very", "get", "got", "one", "two", "like", "go", "use", "used",
    "amp", "re", "s", "t", "m", "bts", "mrt", "skytrain", "bangkok", "thailand",
    "thai", "http", "https", "www", "com", "net", "org",
    "well", "maintained", "really", "quite", "still", "even", "much", "many",
    "way", "time", "day", "year", "place", "area", "people", "thing", "things",
    "make", "made", "take", "took", "come", "came", "know", "think", "want",
    "need", "see", "say", "said", "look", "looks", "feel", "feels", "felt",
}
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def _tokenize(text: str) -> list[str]:
    clean = _URL_RE.sub(" ", str(text))
    return [t for t in re.findall(r"\b[a-z]{3,}\b", clean.lower()) if t not in _STOPWORDS]


def _top_bigrams(texts: list[str], n: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = _tokenize(text)
        for i in range(len(tokens) - 1):
            counter[f"{tokens[i]} {tokens[i+1]}"] += 1
    return [ng for ng, _ in counter.most_common(n)]


def _nss_from(df: pd.DataFrame, sentiment_col: str) -> float:
    n = len(df)
    if n == 0:
        return 0.0
    pos = (df[sentiment_col] == "Positive").sum()
    neg = (df[sentiment_col] == "Negative").sum()
    return float((pos - neg) / n * 100)


def _trend_direction(df: pd.DataFrame, sentiment_col: str) -> TrendDirection:
    df = df.copy()
    df["ym"] = df["created_at_date"].dt.to_period("M")
    months = sorted(df["ym"].unique())
    if len(months) < 2:
        nss = _nss_from(df, sentiment_col)
        return TrendDirection(direction="stable", recent_nss=nss, prior_nss=nss, delta=0.0)
    mid = len(months) // 2
    recent_months = set(months[mid:])
    prior_months = set(months[:mid])
    recent_nss = _nss_from(df[df["ym"].isin(recent_months)], sentiment_col)
    prior_nss = _nss_from(df[df["ym"].isin(prior_months)], sentiment_col)
    delta = recent_nss - prior_nss
    if delta > 5:
        direction = "improving"
    elif delta < -5:
        direction = "declining"
    else:
        direction = "stable"
    return TrendDirection(direction=direction, recent_nss=round(recent_nss, 1), prior_nss=round(prior_nss, 1), delta=round(delta, 1))


def _priority(nss: float, rows: int, trend: TrendDirection) -> str:
    if nss < -20 and rows >= 500:
        return "urgent"
    if nss >= 60 and rows >= 1000:
        return "leverage"
    return "monitor"


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


def _headline(overall_nss: float, top_pain: str, top_strength: str, trend: str) -> str:
    trend_str = {"improving": "improving", "declining": "under pressure", "stable": "holding steady"}[trend]
    return (
        f"Overall sentiment is {trend_str} (NSS {overall_nss:+.1f}%). "
        f"{top_strength} is a standout strength; {top_pain} demands immediate action."
    )


def _filters_from_params(p) -> Filters:
    date_range = None
    if p.date_start and p.date_end:
        date_range = (pd.Timestamp(p.date_start), pd.Timestamp(p.date_end))
    return Filters(
        date_range=date_range,
        aspects=p.aspects or [],
        sources=p.sources or [],
        bts_lines=p.bts_lines or [],
        sentiments=p.sentiments or [],
        min_confidence=p.min_confidence,
        label_source=p.label_source,
    )


def get_business(params) -> BusinessResponse:
    filters = _filters_from_params(params)
    df = apply_filters(load_dashboard_df(), filters)
    sentiment_col = sentiment_col_for(filters)
    aspect_col = "aspect_pred" if filters.label_source == "predicted" else "aspect"

    if df.empty:
        return BusinessResponse(
            summary=ExecutiveSummary(overall_nss=0, total_reviews=0, date_range="—",
                                     top_pain="—", top_strength="—", trend_direction="stable", headline="No data."),
            aspect_deep_dives=[], source_breakdown=[], yoy_nss=[],
        )

    overall_nss = round(_nss_from(df, sentiment_col), 2)
    overall_trend = _trend_direction(df, sentiment_col)
    tbl = aspect_nss_table(df, aspect_col=aspect_col, sentiment_col=sentiment_col)

    pain_row = tbl[tbl["nss"] < 0].sort_values("nss").iloc[0] if not tbl[tbl["nss"] < 0].empty else None
    strength_row = tbl[tbl["nss"] > 0].sort_values("nss", ascending=False).iloc[0] if not tbl[tbl["nss"] > 0].empty else None
    top_pain = str(pain_row["aspect"]) if pain_row is not None else "—"
    top_strength = str(strength_row["aspect"]) if strength_row is not None else "—"

    summary = ExecutiveSummary(
        overall_nss=overall_nss,
        total_reviews=len(df),
        date_range=f"{df['created_at_date'].min().date()} → {df['created_at_date'].max().date()}",
        top_pain=top_pain,
        top_strength=top_strength,
        trend_direction=overall_trend.direction,
        headline=_headline(overall_nss, top_pain, top_strength, overall_trend.direction),
    )

    # Aspect deep dives
    deep_dives: list[AspectDeepDive] = []
    for _, row in tbl.iterrows():
        asp = str(row["aspect"])
        sub = df[df[aspect_col] == asp]
        neg_texts = sub.loc[sub[sentiment_col] == "Negative", "review_text"].astype(str).head(500).tolist()
        pos_texts = sub.loc[sub[sentiment_col] == "Positive", "review_text"].astype(str).head(500).tolist()
        trend = _trend_direction(sub, sentiment_col)
        priority = _priority(float(row["nss"]), int(row["rows"]), trend)
        deep_dives.append(AspectDeepDive(
            aspect=asp,
            nss=round(float(row["nss"]), 1),
            rows=int(row["rows"]),
            pct_negative=round(float(row["pct_negative"]), 1),
            pct_positive=round(float(row["pos"]) / max(int(row["rows"]), 1) * 100, 1),
            top_complaints=_top_bigrams(neg_texts),
            top_praises=_top_bigrams(pos_texts),
            trend=trend,
            priority=priority,
            action=_ACTIONS.get(asp, f"Run a focused improvement sprint on {asp}."),
        ))
    deep_dives.sort(key=lambda d: ({"urgent": 0, "monitor": 1, "leverage": 2}[d.priority], d.nss))

    # Source breakdown
    src_grp = df.groupby("source").agg(
        rows=(sentiment_col, "size"),
        pos=(sentiment_col, lambda s: (s == "Positive").sum()),
        neg=(sentiment_col, lambda s: (s == "Negative").sum()),
    ).reset_index()
    src_grp["nss"] = (src_grp["pos"] - src_grp["neg"]) / src_grp["rows"].clip(lower=1) * 100
    src_grp["pct"] = src_grp["rows"] / len(df) * 100
    src_grp = src_grp.sort_values("rows", ascending=False).head(8)
    source_breakdown = [
        SourceBreakdown(source=str(r["source"]), rows=int(r["rows"]), nss=round(float(r["nss"]), 1), pct=round(float(r["pct"]), 1))
        for _, r in src_grp.iterrows()
    ]

    # YoY NSS
    yoy = df.groupby("year").agg(
        rows=(sentiment_col, "size"),
        pos=(sentiment_col, lambda s: (s == "Positive").sum()),
        neg=(sentiment_col, lambda s: (s == "Negative").sum()),
    ).reset_index()
    yoy["nss"] = (yoy["pos"] - yoy["neg"]) / yoy["rows"].clip(lower=1) * 100
    yoy_nss = [{"year": int(r["year"]), "nss": round(float(r["nss"]), 1), "rows": int(r["rows"])} for _, r in yoy.iterrows()]

    return BusinessResponse(
        summary=summary,
        aspect_deep_dives=deep_dives,
        source_breakdown=source_breakdown,
        yoy_nss=yoy_nss,
    )
