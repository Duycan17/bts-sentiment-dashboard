"""Data loading, group-aware splitting, and sentence-pair formatting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import config


def _stable_text_id(text: str) -> str:
    """Hash of normalized text — same review across rows gets the same id."""
    normalized = " ".join((text or "").split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def load_data(csv_path: Path = config.CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"review_text", "review_rating", "aspect", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df["aspect"] = df["aspect"].astype(str).str.strip()
    df = df[df["sentiment"].isin(config.SENTIMENT_LABELS)].reset_index(drop=True)
    df = df[df["aspect"].isin(config.CANONICAL_ASPECTS)].reset_index(drop=True)
    df["text_id"] = df["review_text"].astype(str).map(_stable_text_id)
    return df


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def make_splits(
    df: pd.DataFrame,
    *,
    n_splits: int = config.SPLIT_N_FOLDS,
    seed: int = config.RANDOM_SEED,
) -> Splits:
    """
    Leakage-safe split: group by text_id, stratify by sentiment.
    fold0 → test, fold1 → val, rest → train.
    """
    y = df["sentiment"].to_numpy()
    groups = df["text_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(df)), y, groups))

    test_idx = folds[0][1]
    val_idx = folds[1][1]
    test_ids = set(df.iloc[test_idx]["text_id"].tolist())
    val_ids = set(df.iloc[val_idx]["text_id"].tolist()) - test_ids

    is_test = df["text_id"].isin(test_ids)
    is_val = df["text_id"].isin(val_ids)

    train = df[~is_test & ~is_val].reset_index(drop=True)
    val = df[is_val].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)

    _verify_no_leakage(train, val, test)
    return Splits(train=train, val=val, test=test)


def _verify_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    tr, va, te = set(train["text_id"]), set(val["text_id"]), set(test["text_id"])
    overlap = (tr & va) | (tr & te) | (va & te)
    if overlap:
        raise RuntimeError(f"Group leakage in splits: {len(overlap)} overlapping text_ids")


def format_pair(aspect: str, text: str) -> str:
    """Sentence-pair encoding for ABSA models."""
    return f"Aspect: {aspect}\nReview: {text}"


def build_xy(df: pd.DataFrame, *, aspect_col: str = "aspect") -> tuple[np.ndarray, np.ndarray]:
    x = df.apply(lambda r: format_pair(r[aspect_col], r["review_text"]), axis=1).to_numpy()
    y = df["sentiment"].to_numpy()
    return x, y
