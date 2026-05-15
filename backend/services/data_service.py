"""Data service — thin wrappers over src/dashboard/* with no Streamlit dependency."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]

# Import from the now-decoupled dashboard helpers
from src.dashboard.data import (
    Filters,
    apply_filters,
    aspect_col_for,
    aspect_nss_table,
    compute_kpis,
    load_dashboard_df,
    sentiment_col_for,
)
from src.dashboard.insights import generate_insight_cards
from src.dashboard.wordcloud import render_wordcloud_png

from backend.models import (
    AspectF1Row,
    AspectNSSResponse,
    AspectRow,
    CalibrationBin,
    ClassMetrics,
    ConfusionMatrixResponse,
    InsightCardResponse,
    InsightsResponse,
    KPIsResponse,
    MetaResponse,
    MonthlyRow,
    NgramRow,
    PerformanceResponse,
    SeasonalityCell,
    TrendDirection,
    TrendResponse,
    VoiceResponse,
)

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "is", "it", "i", "you", "we", "they", "this", "that", "was", "are",
    "be", "have", "has", "had", "do", "did", "not", "from", "by", "as", "so",
    "if", "my", "me", "he", "she", "his", "her", "our", "your", "their", "its",
    "will", "would", "can", "could", "should", "there", "here", "what", "which",
    "who", "how", "when", "where", "why", "all", "any", "some", "no", "more",
    "also", "just", "up", "out", "about", "than", "then", "been", "were", "am",
    "into", "very", "get", "got", "one", "two", "like", "go", "use", "used",
    "amp", "re", "s", "t", "m", "bts", "mrt", "skytrain", "bangkok", "thailand", "thai",
    # extra noise
    "well", "maintained", "really", "quite", "still", "even", "much", "many",
    "way", "time", "day", "year", "place", "area", "people", "thing", "things",
    "make", "made", "take", "took", "come", "came", "know", "think", "want",
    "need", "see", "say", "said", "look", "looks", "feel", "feels", "felt",
    "www", "http", "https", "com", "net", "org", "html", "php", "utm",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def _tokenize(text: str) -> list[str]:
    # Strip URLs before tokenizing
    clean = _URL_RE.sub(" ", str(text))
    return [t for t in re.findall(r"\b[a-z]{3,}\b", clean.lower()) if t not in _STOPWORDS]


def _df() -> pd.DataFrame:
    return load_dashboard_df()


def _filtered(filters: Filters) -> pd.DataFrame:
    return apply_filters(_df(), filters)


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


def get_meta() -> MetaResponse:
    df = _df()
    return MetaResponse(
        sources=sorted(df["source"].dropna().unique().tolist()),
        bts_lines=sorted(df["bts_line"].dropna().unique().tolist()),
        aspects=sorted(df["aspect"].dropna().unique().tolist()),
        date_min=df["created_at_date"].min().date().isoformat(),
        date_max=df["created_at_date"].max().date().isoformat(),
    )


def get_kpis(params) -> KPIsResponse:
    filters = _filters_from_params(params)
    df = _filtered(filters)
    sentiment_col = sentiment_col_for(filters)
    k = compute_kpis(df, sentiment_col=sentiment_col)
    return KPIsResponse(
        rows=k["rows"],
        nss=round(k["nss"], 2),
        pct_positive=round(k["pct_positive"], 2),
        pct_neutral=round(k["pct_neutral"], 2),
        pct_negative=round(k["pct_negative"], 2),
        n_aspects=k["n_aspects"],
        n_sources=k["n_sources"],
        date_min=k["date_min"].date().isoformat() if k["rows"] else "",
        date_max=k["date_max"].date().isoformat() if k["rows"] else "",
        avg_confidence=round(k["avg_confidence"], 4),
    )


def get_aspects(params) -> AspectNSSResponse:
    filters = _filters_from_params(params)
    df = _filtered(filters)
    tbl = aspect_nss_table(df, aspect_col=aspect_col_for(filters), sentiment_col=sentiment_col_for(filters))
    rows = [
        AspectRow(
            aspect=str(r["aspect"]),
            rows=int(r["rows"]),
            pos=int(r["pos"]),
            neg=int(r["neg"]),
            neu=int(r["neu"]),
            nss=round(float(r["nss"]), 2),
            pct_negative=round(float(r["pct_negative"]), 2),
        )
        for _, r in tbl.iterrows()
    ]
    return AspectNSSResponse(aspects=rows)


def get_trends(params) -> TrendResponse:
    filters = _filters_from_params(params)
    df = _filtered(filters)
    sentiment_col = sentiment_col_for(filters)

    if df.empty:
        return TrendResponse(monthly=[], seasonality=[])

    g = df.groupby("year_month").agg(
        rows=(sentiment_col, "size"),
        pos=(sentiment_col, lambda s: (s == "Positive").sum()),
        neg=(sentiment_col, lambda s: (s == "Negative").sum()),
    ).reset_index().sort_values("year_month")
    g["nss"] = (g["pos"] - g["neg"]) / g["rows"].clip(lower=1) * 100

    monthly = [MonthlyRow(year_month=r["year_month"], rows=int(r["rows"]), nss=round(float(r["nss"]), 2)) for _, r in g.iterrows()]

    sg = df.groupby(["year", "month"]).size().reset_index(name="rows")
    seasonality = [SeasonalityCell(year=int(r["year"]), month=int(r["month"]), rows=int(r["rows"])) for _, r in sg.iterrows()]

    return TrendResponse(monthly=monthly, seasonality=seasonality)


def get_voice(params) -> VoiceResponse:
    filters = _filters_from_params(params)
    df = _filtered(filters)
    sentiment_col = sentiment_col_for(filters)

    pos_texts = df.loc[df[sentiment_col] == "Positive", "review_text"].astype(str).head(2000).tolist()
    neg_texts = df.loc[df[sentiment_col] == "Negative", "review_text"].astype(str).head(2000).tolist()

    def top_ngrams(texts: list[str], n: int = 20) -> list[NgramRow]:
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = _tokenize(text)
            for i in range(len(tokens) - 1):
                counter[f"{tokens[i]} {tokens[i+1]}"] += 1
        return [NgramRow(ngram=ng, count=c) for ng, c in counter.most_common(n)]

    return VoiceResponse(
        positive_ngrams=top_ngrams(pos_texts),
        negative_ngrams=top_ngrams(neg_texts),
    )


def get_wordcloud_png(polarity: str, params) -> bytes:
    filters = _filters_from_params(params)
    df = _filtered(filters)
    sentiment_col = sentiment_col_for(filters)
    colormap = "Greens" if polarity == "positive" else "Reds"
    label = "Positive" if polarity == "positive" else "Negative"
    texts = tuple(df.loc[df[sentiment_col] == label, "review_text"].astype(str).head(2000).tolist())
    return render_wordcloud_png(texts, colormap=colormap)


def get_performance(params) -> PerformanceResponse:
    filters = _filters_from_params(params)
    df = _filtered(filters)
    if df.empty:
        empty_cm = ConfusionMatrixResponse(labels=SENTIMENT_LABELS, counts=[[0]*3]*3, normalized=[[0.0]*3]*3)
        return PerformanceResponse(
            accuracy=0, macro_f1=0, weighted_f1=0,
            confusion_matrix=empty_cm, per_class=[], calibration=[],
            aspect_agreement=0, aspect_confusion=empty_cm, per_aspect_f1=[],
        )

    y_true = df["sentiment"].astype(str).to_numpy()
    y_pred = df["sentiment_pred"].astype(str).to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_LABELS)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    report = classification_report(y_true, y_pred, labels=SENTIMENT_LABELS, output_dict=True, zero_division=0)

    per_class = [
        ClassMetrics(
            label=lbl,
            precision=round(report[lbl]["precision"], 4),
            recall=round(report[lbl]["recall"], 4),
            f1=round(report[lbl]["f1-score"], 4),
            support=int(report[lbl]["support"]),
        )
        for lbl in SENTIMENT_LABELS
    ]

    # Calibration
    bins = pd.cut(df["sentiment_confidence"], bins=10, include_lowest=True)
    correct = (df["sentiment"] == df["sentiment_pred"]).astype(int)
    cal_g = pd.DataFrame({"bin": bins, "correct": correct, "conf": df["sentiment_confidence"]}).groupby("bin", observed=True).agg(
        accuracy=("correct", "mean"), avg_conf=("conf", "mean"), rows=("correct", "size"),
    ).reset_index().dropna()
    calibration = [CalibrationBin(avg_conf=round(float(r["avg_conf"]), 4), accuracy=round(float(r["accuracy"]), 4), rows=int(r["rows"])) for _, r in cal_g.iterrows()]

    # Aspect agreement
    aspect_agree = float((df["aspect"] == df["aspect_pred"]).mean())
    all_aspects = sorted(set(df["aspect"]) | set(df["aspect_pred"]))
    acm = confusion_matrix(df["aspect"], df["aspect_pred"], labels=all_aspects)
    acm_norm = acm.astype(float) / acm.sum(axis=1, keepdims=True).clip(min=1)

    # Per-aspect F1
    per_aspect = []
    for asp, sub in df.groupby("aspect"):
        if len(sub) < 30:
            continue
        rep = classification_report(sub["sentiment"], sub["sentiment_pred"], output_dict=True, zero_division=0)
        per_aspect.append(AspectF1Row(
            aspect=str(asp), rows=len(sub),
            accuracy=round(rep["accuracy"], 4),
            macro_f1=round(rep["macro avg"]["f1-score"], 4),
            weighted_f1=round(rep["weighted avg"]["f1-score"], 4),
        ))

    return PerformanceResponse(
        accuracy=round(float(report["accuracy"]), 4),
        macro_f1=round(float(report["macro avg"]["f1-score"]), 4),
        weighted_f1=round(float(report["weighted avg"]["f1-score"]), 4),
        confusion_matrix=ConfusionMatrixResponse(
            labels=SENTIMENT_LABELS,
            counts=cm.tolist(),
            normalized=[[round(v, 4) for v in row] for row in cm_norm.tolist()],
        ),
        per_class=per_class,
        calibration=calibration,
        aspect_agreement=round(aspect_agree, 4),
        aspect_confusion=ConfusionMatrixResponse(
            labels=all_aspects,
            counts=acm.tolist(),
            normalized=[[round(v, 4) for v in row] for row in acm_norm.tolist()],
        ),
        per_aspect_f1=sorted(per_aspect, key=lambda x: x.weighted_f1, reverse=True),
    )


def get_aspect_detail(aspect: str, params) -> "AspectDetailResponse":
    from backend.models import (
        AspectDetailResponse, SampleReview, YearlyNSS, SourceBreakdown
    )
    from backend.services.business_service import (
        _nss_from, _trend_direction, _priority, _top_bigrams, _ACTIONS
    )
    filters = _filters_from_params(params)
    df = apply_filters(load_dashboard_df(), filters)
    sentiment_col = sentiment_col_for(filters)
    aspect_col = aspect_col_for(filters)
    sub = df[df[aspect_col] == aspect].copy()

    if sub.empty:
        return AspectDetailResponse(
            aspect=aspect, nss=0, rows=0, pct_positive=0, pct_neutral=0, pct_negative=0,
            priority="monitor", trend=TrendDirection(direction="stable", recent_nss=0, prior_nss=0, delta=0),
            action=_ACTIONS.get(aspect, ""), top_complaints=[], top_praises=[],
            yearly_nss=[], rating_dist={}, source_dist=[], sample_negative=[], sample_positive=[],
        )

    n = len(sub)
    pos = int((sub[sentiment_col] == "Positive").sum())
    neg = int((sub[sentiment_col] == "Negative").sum())
    neu = int((sub[sentiment_col] == "Neutral").sum())
    nss = round(_nss_from(sub, sentiment_col), 1)
    trend = _trend_direction(sub, sentiment_col)
    priority = _priority(nss, n, trend)

    neg_texts = sub.loc[sub[sentiment_col] == "Negative", "review_text"].astype(str).head(500).tolist()
    pos_texts = sub.loc[sub[sentiment_col] == "Positive", "review_text"].astype(str).head(500).tolist()

    # Yearly NSS
    sub["year"] = sub["created_at_date"].dt.year
    yg = sub.groupby("year").agg(
        rows=(sentiment_col, "size"),
        pos=(sentiment_col, lambda s: (s == "Positive").sum()),
        neg=(sentiment_col, lambda s: (s == "Negative").sum()),
        neu=(sentiment_col, lambda s: (s == "Neutral").sum()),
    ).reset_index()
    yg["nss"] = (yg["pos"] - yg["neg"]) / yg["rows"].clip(lower=1) * 100
    # Drop years with too few reviews — they produce unreliable ±100% NSS
    yg = yg[yg["rows"] >= 20]
    yearly_nss = [
        YearlyNSS(year=int(r["year"]), rows=int(r["rows"]), nss=round(float(r["nss"]), 1),
                  pos=int(r["pos"]), neg=int(r["neg"]), neu=int(r["neu"]))
        for _, r in yg.iterrows()
    ]

    # Rating distribution
    rating_dist = {str(k): int(v) for k, v in sub["review_rating"].value_counts().sort_index().items()}

    # Source distribution
    sg = sub.groupby("source").agg(
        rows=(sentiment_col, "size"),
        pos=(sentiment_col, lambda s: (s == "Positive").sum()),
        neg=(sentiment_col, lambda s: (s == "Negative").sum()),
    ).reset_index()
    sg["nss"] = (sg["pos"] - sg["neg"]) / sg["rows"].clip(lower=1) * 100
    sg["pct"] = sg["rows"] / n * 100
    sg = sg.sort_values("rows", ascending=False).head(6)
    source_dist = [
        SourceBreakdown(source=str(r["source"]), rows=int(r["rows"]),
                        nss=round(float(r["nss"]), 1), pct=round(float(r["pct"]), 1))
        for _, r in sg.iterrows()
    ]

    # Sample reviews
    def samples(texts_df: pd.DataFrame, sent: str, k: int = 6) -> list[SampleReview]:
        rows = texts_df[texts_df[sentiment_col] == sent]
        if rows.empty:
            return []
        rows = rows.sample(n=min(k, len(rows)), random_state=42)
        return [
            SampleReview(
                text=str(r["review_text"])[:350],
                rating=int(r.get("review_rating", 0)),
                source=str(r.get("source", "")),
                sentiment=sent,
                date=str(r["created_at_date"])[:10],
            )
            for _, r in rows.iterrows()
        ]

    return AspectDetailResponse(
        aspect=aspect, nss=nss, rows=n,
        pct_positive=round(pos / n * 100, 1),
        pct_neutral=round(neu / n * 100, 1),
        pct_negative=round(neg / n * 100, 1),
        priority=priority,
        trend=trend,
        action=_ACTIONS.get(aspect, f"Run a focused improvement sprint on {aspect}."),
        top_complaints=_top_bigrams(neg_texts, 6),
        top_praises=_top_bigrams(pos_texts, 6),
        yearly_nss=yearly_nss,
        rating_dist=rating_dist,
        source_dist=source_dist,
        sample_negative=samples(sub, "Negative"),
        sample_positive=samples(sub, "Positive"),
    )


def get_aspect_reviews(aspect: str, params, n: int = 8) -> "AspectReviewsResponse":
    from backend.models import AspectReviewsResponse, SampleReview
    filters = _filters_from_params(params)
    df = _filtered(filters)
    sentiment_col = sentiment_col_for(filters)
    aspect_col = aspect_col_for(filters)
    sub = df[df[aspect_col] == aspect]

    def to_samples(rows: pd.DataFrame) -> list[SampleReview]:
        out = []
        for _, r in rows.sample(n=min(n, len(rows)), random_state=42).iterrows():
            out.append(SampleReview(
                text=str(r["review_text"])[:400],
                rating=int(r.get("review_rating", 0)),
                source=str(r.get("source", "")),
                sentiment=str(r[sentiment_col]),
                date=str(r["created_at_date"])[:10],
            ))
        return out

    neg = sub[sub[sentiment_col] == "Negative"]
    pos = sub[sub[sentiment_col] == "Positive"]
    return AspectReviewsResponse(
        aspect=aspect,
        negative=to_samples(neg),
        positive=to_samples(pos),
    )



    filters = _filters_from_params(params)
    df = _filtered(filters)
    cards = generate_insight_cards(df, aspect_col=aspect_col_for(filters), sentiment_col=sentiment_col_for(filters))
    return InsightsResponse(cards=[
        InsightCardResponse(title=c.title, finding=c.finding, recommendation=c.recommendation, severity=c.severity, metric=c.metric)
        for c in cards
    ])
