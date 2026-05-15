"""Metrics, classification reports, error analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src import config


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = config.SENTIMENT_LABELS
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "name": name,
        "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


def evaluate_by_aspect(
    df: pd.DataFrame,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    aspect_col: str = "aspect",
    min_rows: int = 30,
) -> dict:
    out: dict[str, dict] = {}
    df = df.reset_index(drop=True)
    for aspect, sub in df.groupby(aspect_col):
        if len(sub) < min_rows:
            continue
        idx = sub.index.to_numpy()
        yt = y_true[idx]
        yp = y_pred[idx]
        out[str(aspect)] = {
            "rows": int(len(sub)),
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "report": classification_report(
                yt, yp, labels=config.SENTIMENT_LABELS, output_dict=True, zero_division=0
            ),
        }
    return out


def error_analysis(
    df: pd.DataFrame,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 30,
) -> pd.DataFrame:
    """Return up to n misclassified rows with truncated text."""
    df = df.reset_index(drop=True).copy()
    mask = y_true != y_pred
    errs = df.loc[mask].copy()
    errs["y_true"] = y_true[mask]
    errs["y_pred"] = y_pred[mask]
    errs["text_preview"] = errs["review_text"].astype(str).str.slice(0, 180)
    return errs[["aspect", "y_true", "y_pred", "text_preview"]].head(n)
