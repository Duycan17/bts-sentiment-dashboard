"""Plotly chart factories. Every chart is a pure function: df → go.Figure."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

from src.dashboard.theme import SENTIMENT_COLORS, nss_color
from src.dashboard.data import aspect_nss_table

SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]

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
    "thai",
    # URL fragments + noise
    "http", "https", "www", "com", "net", "org", "html", "php", "utm",
    # common filler
    "well", "maintained", "really", "quite", "still", "even", "much", "many",
    "way", "time", "day", "year", "place", "area", "people", "thing", "things",
    "make", "made", "take", "took", "come", "came", "know", "think", "want",
    "need", "see", "say", "said", "look", "looks", "feel", "feels", "felt",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def _tokenize(text: str) -> list[str]:
    clean = _URL_RE.sub(" ", str(text))
    return [t for t in re.findall(r"\b[a-z]{3,}\b", clean.lower()) if t not in _STOPWORDS]


def _empty_fig(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 14, "color": "#9E9E9E"})
    fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, height=260)
    return fig


# ─── Overview ────────────────────────────────────────────────────────────────

def sentiment_donut(df: pd.DataFrame, *, sentiment_col: str = "sentiment") -> go.Figure:
    counts = df[sentiment_col].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.55,
            marker_colors=[SENTIMENT_COLORS[s] for s in counts.index],
            textinfo="label+percent",
            sort=False,
        )
    )
    fig.update_layout(title="Sentiment mix", showlegend=False, height=320, margin=dict(t=50, b=20))
    return fig


def rating_sentiment_heatmap(df: pd.DataFrame, *, sentiment_col: str = "sentiment") -> go.Figure:
    if df.empty:
        return _empty_fig()
    ct = pd.crosstab(df["review_rating"], df[sentiment_col]).reindex(columns=SENTIMENT_ORDER, fill_value=0)
    fig = px.imshow(
        ct.values,
        x=ct.columns,
        y=[f"★ {r}" for r in ct.index],
        text_auto=True,
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig.update_layout(title="Rating × sentiment", height=320, margin=dict(t=50, b=20))
    return fig


def source_mix(df: pd.DataFrame, *, top_n: int = 10) -> go.Figure:
    counts = df["source"].value_counts().head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color="#3F51B5"))
    fig.update_layout(title=f"Top {top_n} sources", height=380, margin=dict(t=50, b=20, l=140))
    return fig


def line_mix(df: pd.DataFrame) -> go.Figure:
    counts = df["bts_line"].value_counts()
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color="#00BCD4"))
    fig.update_layout(title="Reviews by BTS line", height=320, margin=dict(t=50, b=20))
    return fig


# ─── Aspect Pulse ────────────────────────────────────────────────────────────

def nss_by_aspect_bar(df: pd.DataFrame, *, aspect_col: str = "aspect", sentiment_col: str = "sentiment") -> go.Figure:
    tbl = aspect_nss_table(df, aspect_col=aspect_col, sentiment_col=sentiment_col)
    if tbl.empty:
        return _empty_fig()
    colors = [nss_color(v) for v in tbl["nss"]]
    fig = go.Figure(
        go.Bar(
            x=tbl["nss"],
            y=tbl["aspect"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in tbl["nss"]],
            textposition="outside",
            customdata=tbl[["rows", "pct_negative"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>NSS: %{x:.2f}%<br>"
                "Volume: %{customdata[0]} reviews<br>"
                "Negative share: %{customdata[1]:.1f}%<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(
        title="Net Sentiment Score (NSS) by aspect",
        xaxis_title="NSS (%)",
        height=480,
        margin=dict(t=50, b=20, l=200, r=80),
    )
    return fig


def aspect_sentiment_heatmap(df: pd.DataFrame, *, aspect_col: str = "aspect", sentiment_col: str = "sentiment") -> go.Figure:
    if df.empty:
        return _empty_fig()
    ct = pd.crosstab(df[aspect_col], df[sentiment_col]).reindex(columns=SENTIMENT_ORDER, fill_value=0)
    pct = ct.div(ct.sum(axis=1).clip(lower=1), axis=0) * 100
    fig = px.imshow(
        pct.values,
        x=pct.columns,
        y=pct.index,
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=100,
        text_auto=".1f",
        aspect="auto",
    )
    fig.update_layout(title="Aspect × sentiment (% within aspect)", height=480, margin=dict(t=50, b=20, l=200))
    return fig


def volume_nss_quadrant(df: pd.DataFrame, *, aspect_col: str = "aspect", sentiment_col: str = "sentiment") -> go.Figure:
    tbl = aspect_nss_table(df, aspect_col=aspect_col, sentiment_col=sentiment_col)
    if tbl.empty:
        return _empty_fig()
    median_vol = tbl["rows"].median()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tbl["rows"],
            y=tbl["nss"],
            mode="markers+text",
            marker=dict(
                size=tbl["rows"] / tbl["rows"].max() * 50 + 12,
                color=[nss_color(v) for v in tbl["nss"]],
                line=dict(width=1, color="white"),
            ),
            text=tbl["aspect"],
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>Volume: %{x}<br>NSS: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="#666", line_dash="dash")
    fig.add_vline(x=median_vol, line_color="#666", line_dash="dash")
    fig.add_annotation(x=tbl["rows"].max(), y=tbl["nss"].min(), text="High-volume pain →",
                       showarrow=False, font=dict(color="#F44336", size=11), xanchor="right", yanchor="bottom")
    fig.add_annotation(x=tbl["rows"].max(), y=tbl["nss"].max(), text="High-volume strengths",
                       showarrow=False, font=dict(color="#4CAF50", size=11), xanchor="right", yanchor="top")
    fig.update_layout(
        title="Volume vs. NSS — where to invest",
        xaxis_title="Review volume",
        yaxis_title="NSS (%)",
        height=520,
        margin=dict(t=50, b=20),
    )
    return fig


# ─── Trends ──────────────────────────────────────────────────────────────────

def monthly_volume_line(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    g = df.groupby("year_month").size().reset_index(name="rows").sort_values("year_month")
    fig = go.Figure(go.Scatter(x=g["year_month"], y=g["rows"], mode="lines+markers", line=dict(color="#3F51B5")))
    fig.update_layout(title="Monthly review volume", xaxis_title="Month", yaxis_title="Reviews", height=320)
    return fig


def monthly_nss_line(df: pd.DataFrame, *, sentiment_col: str = "sentiment") -> go.Figure:
    if df.empty:
        return _empty_fig()
    g = df.groupby("year_month").agg(
        rows=(sentiment_col, "size"),
        pos=(sentiment_col, lambda s: (s == "Positive").sum()),
        neg=(sentiment_col, lambda s: (s == "Negative").sum()),
    ).reset_index()
    g["nss"] = (g["pos"] - g["neg"]) / g["rows"].clip(lower=1) * 100
    g = g.sort_values("year_month")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["year_month"], y=g["nss"], mode="lines+markers", line=dict(color="#4CAF50"), name="NSS"))
    fig.add_hline(y=0, line_color="#F44336", line_dash="dash")
    fig.update_layout(title="Monthly Net Sentiment Score", xaxis_title="Month", yaxis_title="NSS (%)", height=320)
    return fig


def nss_by_aspect_smallmult(df: pd.DataFrame, *, aspect_col: str = "aspect", sentiment_col: str = "sentiment") -> go.Figure:
    if df.empty:
        return _empty_fig()
    g = df.groupby(["year_month", aspect_col]).agg(
        rows=(sentiment_col, "size"),
        pos=(sentiment_col, lambda s: (s == "Positive").sum()),
        neg=(sentiment_col, lambda s: (s == "Negative").sum()),
    ).reset_index()
    g["nss"] = (g["pos"] - g["neg"]) / g["rows"].clip(lower=1) * 100
    g = g[g["rows"] >= 5]
    if g.empty:
        return _empty_fig("Not enough data per aspect/month")
    fig = px.line(g, x="year_month", y="nss", facet_col=aspect_col, facet_col_wrap=3, height=620)
    fig.update_yaxes(title_text="")
    fig.update_xaxes(title_text="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(title="NSS over time, per aspect", margin=dict(t=60, b=20))
    return fig


def seasonality_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    g = df.groupby(["year", "month"]).size().reset_index(name="rows")
    pivot = g.pivot(index="year", columns="month", values="rows").fillna(0)
    pivot = pivot.reindex(columns=range(1, 13), fill_value=0)
    fig = px.imshow(
        pivot.values,
        x=[f"M{m:02d}" for m in pivot.columns],
        y=[str(int(y)) for y in pivot.index],
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=True,
    )
    fig.update_layout(title="Volume seasonality (year × month)", height=380, margin=dict(t=50, b=20))
    return fig


# ─── Voice of Customer ───────────────────────────────────────────────────────

def top_ngrams_bar(texts: list[str], *, n: int = 20, ngram: tuple[int, int] = (2, 2), title: str = "Top n-grams", color: str = "#3F51B5") -> go.Figure:
    tokens_per_doc = [_tokenize(t) for t in texts]
    counter: Counter[str] = Counter()
    lo, hi = ngram
    for tokens in tokens_per_doc:
        for k in range(lo, hi + 1):
            for i in range(len(tokens) - k + 1):
                counter[" ".join(tokens[i : i + k])] += 1
    top = counter.most_common(n)
    if not top:
        return _empty_fig()
    labels, counts = zip(*top)
    fig = go.Figure(go.Bar(x=list(counts)[::-1], y=list(labels)[::-1], orientation="h", marker_color=color))
    fig.update_layout(title=title, height=480, margin=dict(t=50, b=20, l=160))
    return fig


# ─── Model Performance ──────────────────────────────────────────────────────

def confusion_matrix_fig(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[str] = SENTIMENT_ORDER, title: str = "Confusion matrix") -> go.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Counts", "Normalized (per row)"))
    fig.add_trace(
        go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues", showscale=False,
                   text=cm, texttemplate="%{text}", hovertemplate="True %{y} → Pred %{x}: %{z}<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=cm_norm, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1,
                   text=cm_norm, texttemplate="%{text:.2f}", hovertemplate="True %{y} → Pred %{x}: %{z:.2%}<extra></extra>"),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="True", autorange="reversed")
    fig.update_layout(title=title, height=420, margin=dict(t=70, b=20))
    return fig


def per_class_metrics_bar(report: dict, *, labels: list[str] = SENTIMENT_ORDER, title: str = "Per-class metrics") -> go.Figure:
    metrics = ["precision", "recall", "f1-score"]
    fig = go.Figure()
    for m in metrics:
        fig.add_trace(go.Bar(name=m, x=labels, y=[report[l][m] for l in labels]))
    fig.update_layout(barmode="group", title=title, yaxis=dict(range=[0, 1]), height=380)
    return fig


def confidence_calibration(df: pd.DataFrame, *, n_bins: int = 10) -> go.Figure:
    if df.empty:
        return _empty_fig()
    bins = pd.cut(df["sentiment_confidence"], bins=n_bins, include_lowest=True)
    correct = (df["sentiment"] == df["sentiment_pred"]).astype(int)
    g = pd.DataFrame({"bin": bins, "correct": correct, "conf": df["sentiment_confidence"]}).groupby("bin", observed=True).agg(
        accuracy=("correct", "mean"),
        avg_conf=("conf", "mean"),
        rows=("correct", "size"),
    ).reset_index().dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#9E9E9E"), name="Ideal"))
    fig.add_trace(
        go.Scatter(
            x=g["avg_conf"], y=g["accuracy"], mode="markers+lines",
            marker=dict(size=g["rows"] / max(g["rows"].max(), 1) * 30 + 6, color="#3F51B5"),
            name="Observed", customdata=g[["rows"]].values,
            hovertemplate="conf bin avg=%{x:.2f}<br>accuracy=%{y:.2%}<br>n=%{customdata[0]}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Confidence calibration — predicted vs. observed accuracy",
        xaxis_title="Predicted confidence", yaxis_title="Observed accuracy",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        height=420, margin=dict(t=50, b=20),
    )
    return fig


def aspect_agreement_diagnostic(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    aspects = sorted(set(df["aspect"]) | set(df["aspect_pred"]))
    cm = confusion_matrix(df["aspect"], df["aspect_pred"], labels=aspects)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig = px.imshow(cm_norm, x=aspects, y=aspects, color_continuous_scale="Blues", zmin=0, zmax=1, text_auto=".2f", aspect="auto")
    fig.update_layout(
        title="Aspect agreement: ground truth (rows) vs. predicted (cols)",
        height=560, margin=dict(t=50, b=20, l=200),
        xaxis=dict(tickangle=35), yaxis=dict(autorange="reversed"),
    )
    return fig
