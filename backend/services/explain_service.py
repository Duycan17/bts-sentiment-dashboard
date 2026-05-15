"""Explain service — wraps src/dashboard/explain.py for FastAPI."""

from __future__ import annotations

from src.dashboard.explain import ExplainResult, explain_baseline, predict_baseline
from backend.models import ExplainResponse, PredictResponse, TokenContribution


def predict(text: str, aspect: str) -> PredictResponse:
    label, proba = predict_baseline(text, aspect)
    return PredictResponse(label=label, confidence=round(proba[label], 4), proba={k: round(v, 4) for k, v in proba.items()})


def explain(text: str, aspect: str, top_k: int = 15) -> ExplainResponse:
    result: ExplainResult = explain_baseline(text, aspect)
    pairs = sorted(zip(result.tokens, result.contributions), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    pairs.sort(key=lambda kv: kv[1])
    return ExplainResponse(
        label=result.predicted_label,
        confidence=round(result.confidence, 4),
        proba={k: round(v, 4) for k, v in result.proba.items()},
        contributions=[TokenContribution(token=t, value=round(v, 4)) for t, v in pairs],
        base_value=round(result.base_value, 4),
    )
