"""Inference CLI: rule-based aspect → ABSA sentiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import config
from src.aspect_extractor import extract_aspect
from src.data import format_pair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict aspect + sentiment for a review.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Single review text.")
    src.add_argument("--file", type=Path, help="Path to a file with one review per line.")
    p.add_argument("--model", choices=["baseline", "transformer"], default="baseline")
    return p.parse_args()


def _load_inputs(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [args.text]
    return [line.strip() for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _predict_baseline(pairs: list[str]) -> tuple[list[str], list[float]]:
    from src.models.baseline import load_baseline

    if not config.BASELINE_PATH.exists():
        sys.exit(f"Baseline model not found at {config.BASELINE_PATH}. Run train.py first.")
    pipe = load_baseline()
    proba = pipe.predict_proba(pairs)
    labels = pipe.classes_
    idx = proba.argmax(axis=1)
    return [str(labels[i]) for i in idx], [float(proba[i, idx[i]]) for i in range(len(idx))]


def _predict_transformer(reviews: list[str], aspects: list[str]) -> tuple[list[str], list[float]]:
    import pandas as pd

    from src.models.transformer import predict_transformer

    if not config.TRANSFORMER_DIR.exists():
        sys.exit(f"Transformer not found at {config.TRANSFORMER_DIR}. Run train.py first.")
    df = pd.DataFrame({"review_text": reviews, "aspect": aspects})
    labels, probs = predict_transformer(df)
    confidences = probs.max(axis=1).tolist()
    return list(labels), [float(c) for c in confidences]


def main() -> None:
    args = parse_args()
    reviews = _load_inputs(args)

    aspects = [extract_aspect(t) for t in reviews]

    if args.model == "baseline":
        pairs = [format_pair(a, t) for a, t in zip(aspects, reviews)]
        sentiments, confidences = _predict_baseline(pairs)
    else:
        sentiments, confidences = _predict_transformer(reviews, aspects)

    output = [
        {
            "text": t[:200],
            "aspect": a,
            "sentiment": s,
            "confidence": round(c, 4),
        }
        for t, a, s, c in zip(reviews, aspects, sentiments, confidences)
    ]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
