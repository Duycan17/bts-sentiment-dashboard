"""SHAP explainability for the saved baseline (TF-IDF + LogReg) pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from src.dashboard.theme import SENTIMENT_COLORS

# Module-level singletons — loaded once per process, thread-safe for read-only use
_pipeline = None
_explainer_obj = None

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
BACKGROUND_PARQUET = ARTIFACTS / "shap_background.parquet"
BACKGROUND_SIZE = 200


@dataclass
class ExplainResult:
    text: str
    aspect: str
    predicted_label: str
    confidence: float
    proba: dict[str, float]
    tokens: list[str]
    contributions: list[float]  # signed, predicted-class
    base_value: float


def _load_pipeline():
    """Load the joblib baseline pipeline once per process (module-level singleton)."""
    global _pipeline
    if _pipeline is None:
        from src.models.baseline import load_baseline
        _pipeline = load_baseline()
    return _pipeline


def _format_pair(aspect: str, text: str) -> str:
    """Match the training-time encoding used by the baseline."""
    return f"Aspect: {aspect}\nReview: {text}"


def _build_or_load_background(pipeline) -> np.ndarray:
    """Sample BACKGROUND_SIZE training pairs once and cache to parquet."""
    tfidf = pipeline.named_steps["tfidf"]
    if BACKGROUND_PARQUET.exists():
        df = pd.read_parquet(BACKGROUND_PARQUET)
        pairs = df["pair"].astype(str).tolist()
        return tfidf.transform(pairs).toarray()

    src_csv = ROOT / "DATA_QUALITY_CHECKLIST_filled.csv"
    src = pd.read_csv(src_csv, usecols=["review_text", "aspect"]).dropna()
    sample = src.sample(n=min(BACKGROUND_SIZE, len(src)), random_state=42)
    pairs = [_format_pair(a, t) for a, t in zip(sample["aspect"], sample["review_text"])]
    pd.DataFrame({"pair": pairs}).to_parquet(BACKGROUND_PARQUET, index=False)
    return tfidf.transform(pairs).toarray()


def _explainer():
    """Build SHAP explainer once per process (module-level singleton)."""
    global _explainer_obj
    if _explainer_obj is None:
        pipeline = _load_pipeline()
        background = _build_or_load_background(pipeline)
        logreg = pipeline.named_steps["logreg"]
        masker = shap.maskers.Independent(background)
        _explainer_obj = (pipeline, shap.LinearExplainer(logreg, masker=masker))
    return _explainer_obj


def predict_baseline(text: str, aspect: str) -> tuple[str, dict[str, float]]:
    pipeline = _load_pipeline()
    pair = _format_pair(aspect, text)
    proba = pipeline.predict_proba([pair])[0]
    classes = list(pipeline.classes_)
    proba_dict = {c: float(p) for c, p in zip(classes, proba)}
    label = classes[int(np.argmax(proba))]
    return label, proba_dict


def explain_baseline(text: str, aspect: str) -> ExplainResult:
    pipeline, explainer = _explainer()
    tfidf = pipeline.named_steps["tfidf"]
    classes = list(pipeline.classes_)

    pair = _format_pair(aspect, text)
    vec = tfidf.transform([pair])
    proba = pipeline.predict_proba([pair])[0]
    proba_dict = {c: float(p) for c, p in zip(classes, proba)}
    label = classes[int(np.argmax(proba))]
    label_idx = int(np.argmax(proba))

    sv = explainer(vec)  # shap.Explanation, shape (1, n_features, n_classes) for multiclass linear
    values = sv.values[0]
    if values.ndim == 2:
        contributions_full = values[:, label_idx]
        base_value = float(np.atleast_1d(sv.base_values[0])[label_idx])
    else:
        contributions_full = values
        base_value = float(np.atleast_1d(sv.base_values[0])[0])

    feature_names = tfidf.get_feature_names_out()
    nz = vec.nonzero()[1]
    tokens = [str(feature_names[i]) for i in nz]
    contribs = [float(contributions_full[i]) for i in nz]

    return ExplainResult(
        text=text,
        aspect=aspect,
        predicted_label=label,
        confidence=float(proba[label_idx]),
        proba=proba_dict,
        tokens=tokens,
        contributions=contribs,
        base_value=base_value,
    )


def render_token_contributions(result: ExplainResult, *, top_k: int = 15) -> go.Figure:
    if not result.tokens:
        fig = go.Figure()
        fig.add_annotation(text="No vocabulary tokens found in the input.", showarrow=False)
        fig.update_layout(height=260)
        return fig
    pairs = sorted(zip(result.tokens, result.contributions), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    pairs.sort(key=lambda kv: kv[1])
    tokens = [k for k, _ in pairs]
    values = [v for _, v in pairs]
    label = result.predicted_label
    pos_color = SENTIMENT_COLORS.get(label, "#4CAF50")
    neg_color = "#9E9E9E"
    colors = [pos_color if v > 0 else neg_color for v in values]
    fig = go.Figure(
        go.Bar(
            x=values, y=tokens, orientation="h", marker_color=colors,
            text=[f"{v:+.3f}" for v in values], textposition="outside",
            hovertemplate="Token <b>%{y}</b><br>SHAP value: %{x:+.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"Top {top_k} tokens driving '{label}' (sum = log-odds shift from base)",
        xaxis_title="SHAP value (signed contribution)",
        yaxis_title="",
        height=480,
        margin=dict(t=60, b=20, l=140, r=80),
    )
    return fig
