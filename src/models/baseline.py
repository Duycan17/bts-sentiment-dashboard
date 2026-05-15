"""TF-IDF + LogisticRegression baseline (Traditional ML)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.TFIDF_MAX_FEATURES,
                    ngram_range=config.TFIDF_NGRAM,
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    C=config.LOGREG_C,
                    max_iter=config.LOGREG_MAX_ITER,
                    class_weight="balanced",
                    n_jobs=None,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def train_baseline(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    pipe = build_pipeline()
    pipe.fit(x_train, y_train)
    return pipe


def save_baseline(pipe: Pipeline, path: Path = config.BASELINE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return path


def load_baseline(path: Path = config.BASELINE_PATH) -> Pipeline:
    return joblib.load(path)
