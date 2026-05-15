"""Error analysis endpoint — FP, FN, False Neutral, missed Neutral, low-confidence errors."""

from __future__ import annotations

import pandas as pd

from src.dashboard.data import (
    Filters,
    apply_filters,
    load_dashboard_df,
    sentiment_col_for,
    aspect_col_for,
)
from backend.models import SampleReview


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


def _to_sample(r: pd.Series, sentiment_col: str) -> dict:
    return {
        "text": str(r["review_text"])[:350],
        "rating": int(r.get("review_rating", 0)),
        "source": str(r.get("source", "")),
        "sentiment": str(r[sentiment_col]),
        "date": str(r["created_at_date"])[:10],
        "aspect": str(r.get("aspect", "")),
        "predicted": str(r.get("sentiment_pred", "")),
        "confidence": round(float(r.get("sentiment_confidence", 0)), 3),
    }


def get_error_analysis(params) -> dict:
    filters = _filters_from_params(params)
    df = apply_filters(load_dashboard_df(), filters)
    aspect_col = aspect_col_for(filters)

    # Always compare ground truth vs model prediction — independent of label_source filter
    y_true = df["sentiment"].astype(str)
    y_pred = df["sentiment_pred"].astype(str)

    # Error type masks
    fp_mask  = (y_true == "Negative") & (y_pred == "Positive")   # False Positive
    fn_mask  = (y_true == "Positive") & (y_pred == "Negative")   # False Negative
    fnu_mask = (y_true != "Neutral")  & (y_pred == "Neutral")    # False Neutral
    mnu_mask = (y_true == "Neutral")  & (y_pred != "Neutral")    # Missed Neutral
    lce_mask = (y_true != y_pred)     & (df["sentiment_confidence"] < 0.6)  # Low-conf errors

    def samples(mask: pd.Series, n: int = 8) -> list[dict]:
        sub = df[mask]
        if sub.empty:
            return []
        return [_to_sample(r, "sentiment") for _, r in sub.sample(n=min(n, len(sub)), random_state=42).iterrows()]

    # Per-aspect error rates
    aspect_errors = []
    for asp, sub in df.groupby(aspect_col):
        n = len(sub)
        if n < 20:
            continue
        yt = sub["sentiment"].astype(str)
        yp = sub["sentiment_pred"].astype(str)
        errors = int((yt != yp).sum())
        aspect_errors.append({
            "aspect": str(asp),
            "rows": n,
            "errors": errors,
            "error_rate": round(errors / n * 100, 1),
            "fp": int(((yt == "Negative") & (yp == "Positive")).sum()),
            "fn": int(((yt == "Positive") & (yp == "Negative")).sum()),
            "false_neutral": int(((yt != "Neutral") & (yp == "Neutral")).sum()),
        })
    aspect_errors.sort(key=lambda x: x["error_rate"], reverse=True)

    # Confusion matrix counts
    labels = ["Negative", "Neutral", "Positive"]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Overall stats
    total = len(df)
    total_errors = int((y_true != y_pred).sum())

    return {
        "total": total,
        "total_errors": total_errors,
        "error_rate": round(total_errors / max(total, 1) * 100, 1),
        "fp_count": int(fp_mask.sum()),
        "fn_count": int(fn_mask.sum()),
        "false_neutral_count": int(fnu_mask.sum()),
        "missed_neutral_count": int(mnu_mask.sum()),
        "low_conf_error_count": int(lce_mask.sum()),
        "confusion_matrix": {"labels": labels, "counts": cm},
        "fp_samples": samples(fp_mask),
        "fn_samples": samples(fn_mask),
        "false_neutral_samples": samples(fnu_mask),
        "missed_neutral_samples": samples(mnu_mask),
        "low_conf_samples": samples(lce_mask),
        "aspect_errors": aspect_errors,
    }
