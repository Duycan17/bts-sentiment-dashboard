"""Visualizations: confusion matrices, NSS trends, word clouds, topic modeling."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

from src import config

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "i", "you", "we", "they", "this", "that",
    "was", "are", "be", "have", "has", "had", "do", "did", "not", "from",
    "by", "as", "so", "if", "my", "me", "he", "she", "his", "her", "our",
    "your", "their", "its", "will", "would", "can", "could", "should",
    "there", "here", "what", "which", "who", "how", "when", "where", "why",
    "all", "any", "some", "no", "more", "also", "just", "up", "out", "about",
    "than", "then", "been", "were", "am", "into", "very", "get", "got",
    "one", "two", "like", "go", "use", "used", "amp", "re", "s", "t", "m",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-z]{3,}\b", str(text).lower())
    return [t for t in tokens if t not in _STOPWORDS]


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    out_path: Path,
) -> None:
    labels = config.SENTIMENT_LABELS
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"{title} — counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title(f"{title} — normalized")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_nss_trend(df: pd.DataFrame, *, out_path: Path) -> None:
    """Daily NSS line chart from `created_at_date`."""
    work = df.copy()
    work["created_at_date"] = pd.to_datetime(work["created_at_date"], errors="coerce")
    work = work.dropna(subset=["created_at_date"])
    if work.empty:
        return
    work["date"] = work["created_at_date"].dt.date.astype(str)
    grp = work.groupby("date").agg(
        rows=("sentiment", "size"),
        pos=("sentiment", lambda s: (s == "Positive").sum()),
        neg=("sentiment", lambda s: (s == "Negative").sum()),
    )
    grp["nss"] = (grp["pos"] - grp["neg"]) / grp["rows"].clip(lower=1) * 100

    plt.figure(figsize=(12, 4))
    sns.lineplot(data=grp.reset_index(), x="date", y="nss", marker="o")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Daily Net Sentiment Score (NSS)")
    plt.ylabel("NSS (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_nss_by_aspect(df: pd.DataFrame, *, out_path: Path) -> None:
    grp = df.groupby("aspect").agg(
        rows=("sentiment", "size"),
        pos=("sentiment", lambda s: (s == "Positive").sum()),
        neg=("sentiment", lambda s: (s == "Negative").sum()),
    )
    grp["nss"] = (grp["pos"] - grp["neg"]) / grp["rows"].clip(lower=1) * 100
    grp = grp.sort_values("nss")

    plt.figure(figsize=(11, 6))
    colors = ["#F44336" if v < 0 else "#FF9800" if v < 30 else "#4CAF50" for v in grp["nss"]]
    plt.barh(grp.index, grp["nss"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Net Sentiment Score by Aspect")
    plt.xlabel("NSS (%)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_wordcloud(texts: list[str], *, title: str, colormap: str, out_path: Path) -> None:
    tokens: list[str] = []
    for t in texts:
        tokens.extend(_tokenize(t))
    if not tokens:
        return
    freq = Counter(tokens)
    wc = WordCloud(
        width=900, height=450,
        background_color="white",
        colormap=colormap,
        max_words=config.WORDCLOUD_MAX_WORDS,
        collocations=False,
    ).generate_from_frequencies(freq)
    plt.figure(figsize=(11, 5.5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_topics_lda(
    texts: list[str],
    *,
    title: str,
    out_path: Path,
    n_topics: int = config.LDA_N_TOPICS,
    n_top_words: int = config.LDA_N_TOP_WORDS,
) -> None:
    if len(texts) < n_topics * 2:
        return
    vec = CountVectorizer(
        max_features=4000,
        stop_words=list(_STOPWORDS),
        token_pattern=r"\b[a-z]{3,}\b",
        min_df=5,
    )
    X = vec.fit_transform(texts)
    if X.shape[1] == 0:
        return
    feature_names = vec.get_feature_names_out()
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=config.RANDOM_SEED,
        max_iter=15,
        learning_method="batch",
    )
    lda.fit(X)

    fig, axes = plt.subplots(2, (n_topics + 1) // 2, figsize=(14, 7), sharey=False)
    axes = axes.flatten()
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[: -n_top_words - 1 : -1]
        words = [feature_names[j] for j in top_idx]
        weights = comp[top_idx]
        axes[i].barh(words[::-1], weights[::-1], color="#3F51B5")
        axes[i].set_title(f"Topic {i + 1}")
    for j in range(len(lda.components_), len(axes)):
        axes[j].axis("off")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_per_class_pr(report: dict, *, title: str, out_path: Path) -> None:
    """Precision/Recall/F1 bars per class from a classification_report dict."""
    labels = config.SENTIMENT_LABELS
    metrics = ["precision", "recall", "f1-score"]
    data = np.array([[report[l][m] for m in metrics] for l in labels])

    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(9, 5))
    for i, m in enumerate(metrics):
        plt.bar(x + (i - 1) * width, data[:, i], width, label=m)
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
