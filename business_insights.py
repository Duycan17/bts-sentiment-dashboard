from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import absa_pipeline as absa


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def nss(df: pd.DataFrame) -> float:
    total = len(df)
    if total == 0:
        return 0.0
    pos = (df["sentiment"] == "Positive").sum()
    neg = (df["sentiment"] == "Negative").sum()
    return float((pos - neg) / total * 100.0)


def _nss_from_counts(pos: int, neg: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float((pos - neg) / total * 100.0)


def main(args: argparse.Namespace) -> None:
    raw = absa.load_raw(Path(args.csv))
    df = absa.clean_labels(raw)

    df["created_at_date"] = pd.to_datetime(df["created_at_date"], errors="coerce")
    df = df.dropna(subset=["created_at_date"]).copy()

    df["date"] = df["created_at_date"].dt.date.astype(str)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # NSS overall
    overall = {"rows": int(len(df)), "nss": nss(df)}

    df["_is_pos"] = (df["sentiment"] == "Positive").astype(int)
    df["_is_neg"] = (df["sentiment"] == "Negative").astype(int)

    daily = df.groupby("date", as_index=False).agg(
        rows=("sentiment", "size"),
        pos=("_is_pos", "sum"),
        neg=("_is_neg", "sum"),
    )
    daily["nss"] = [
        _nss_from_counts(p, n, t) for p, n, t in zip(daily["pos"], daily["neg"], daily["rows"])
    ]
    daily = daily.drop(columns=["pos", "neg"])

    by_aspect = df.groupby("aspect", as_index=False).agg(
        rows=("sentiment", "size"),
        pos=("_is_pos", "sum"),
        neg=("_is_neg", "sum"),
    )
    by_aspect["nss"] = [
        _nss_from_counts(p, n, t)
        for p, n, t in zip(by_aspect["pos"], by_aspect["neg"], by_aspect["rows"])
    ]
    by_aspect = by_aspect.drop(columns=["pos", "neg"]).sort_values("rows", ascending=False)

    (ARTIFACTS_DIR / "nss_overall.txt").write_text(
        f"rows={overall['rows']}\nNSS={overall['nss']:.2f}\n", encoding="utf-8"
    )
    daily.to_csv(ARTIFACTS_DIR / "nss_daily.csv", index=False)
    by_aspect.to_csv(ARTIFACTS_DIR / "nss_by_aspect.csv", index=False)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=daily, x="date", y="nss")
    plt.xticks(rotation=45, ha="right")
    plt.title("Net Sentiment Score (NSS) over time")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "nss_trend.png", dpi=160)
    plt.close()

    top = by_aspect.head(12).copy()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top, y="aspect", x="nss", orient="h")
    plt.title("NSS by aspect (top aspects by volume)")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "nss_by_aspect.png", dpi=160)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=str(absa.CSV_PATH))
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

