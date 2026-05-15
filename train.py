"""Train + evaluate ABSA pipeline. Saves models, figures, logs, and reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data import Splits, build_xy, load_data, make_splits
from src.evaluate import error_analysis, evaluate, evaluate_by_aspect
from src.models.baseline import save_baseline, train_baseline
from src.models.transformer import predict_transformer, train_transformer
from src.utils import Timer, save_json, set_seeds, setup_logging
from src.visualize import (
    plot_confusion_matrix,
    plot_nss_by_aspect,
    plot_nss_trend,
    plot_per_class_pr,
    plot_topics_lda,
    plot_wordcloud,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ABSA sentiment models.")
    p.add_argument("--csv", type=Path, default=config.CSV_PATH)
    p.add_argument("--quick", action="store_true", help="Smoke run: cap rows + 1 epoch.")
    p.add_argument("--skip-transformer", action="store_true")
    return p.parse_args()


def _save_visualizations(df: pd.DataFrame, log) -> None:
    with Timer("save NSS + wordclouds + topics", log):
        plot_nss_trend(df, out_path=config.FIGURES_DIR / "nss_trend.png")
        plot_nss_by_aspect(df, out_path=config.FIGURES_DIR / "nss_by_aspect.png")

        pos_texts = df.loc[df["sentiment"] == "Positive", "review_text"].astype(str).tolist()
        neg_texts = df.loc[df["sentiment"] == "Negative", "review_text"].astype(str).tolist()

        plot_wordcloud(pos_texts, title="Positive Reviews",
                       colormap="Greens", out_path=config.FIGURES_DIR / "wordcloud_positive.png")
        plot_wordcloud(neg_texts, title="Negative Reviews",
                       colormap="Reds", out_path=config.FIGURES_DIR / "wordcloud_negative.png")

        plot_topics_lda(pos_texts, title="LDA Topics — Positive",
                        out_path=config.FIGURES_DIR / "topic_lda_positive.png")
        plot_topics_lda(neg_texts, title="LDA Topics — Negative",
                        out_path=config.FIGURES_DIR / "topic_lda_negative.png")


def _eval_and_plot(
    name: str,
    df_split: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    cm_path: Path,
    pr_path: Path,
) -> dict:
    metrics = evaluate(name, y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, title=name, out_path=cm_path)
    plot_per_class_pr(metrics["report"], title=name, out_path=pr_path)
    metrics["by_aspect"] = evaluate_by_aspect(df_split, y_true=y_true, y_pred=y_pred)
    return metrics


def main() -> None:
    args = parse_args()
    log = setup_logging(config.LOGS_DIR / "training.log")
    set_seeds(config.RANDOM_SEED)

    log.info(f"Args: csv={args.csv} quick={args.quick} skip_transformer={args.skip_transformer}")

    with Timer("load + split", log):
        df = load_data(args.csv)
        if args.quick:
            df = df.sample(n=min(config.QUICK_MAX_ROWS, len(df)),
                           random_state=config.RANDOM_SEED).reset_index(drop=True)
        splits: Splits = make_splits(df)
        log.info(f"rows: total={len(df)} train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")

    x_train, y_train = build_xy(splits.train)
    x_val, y_val = build_xy(splits.val)
    x_test, y_test = build_xy(splits.test)

    results: dict = {
        "data": {
            "rows_total": int(len(df)),
            "rows_train": int(len(splits.train)),
            "rows_val": int(len(splits.val)),
            "rows_test": int(len(splits.test)),
            "quick_mode": args.quick,
        }
    }

    # ── Baseline ──────────────────────────────────────────────────────────
    with Timer("train baseline (TF-IDF + LogReg)", log):
        baseline = train_baseline(x_train, y_train)
        save_baseline(baseline)

    with Timer("evaluate baseline", log):
        val_pred = baseline.predict(x_val)
        test_pred = baseline.predict(x_test)
        results["baseline"] = {
            "val": _eval_and_plot(
                "baseline_val", splits.val, y_val, val_pred,
                cm_path=config.FIGURES_DIR / "cm_baseline_val.png",
                pr_path=config.FIGURES_DIR / "pr_baseline_val.png",
            ),
            "test": _eval_and_plot(
                "baseline_test", splits.test, y_test, test_pred,
                cm_path=config.FIGURES_DIR / "cm_baseline_test.png",
                pr_path=config.FIGURES_DIR / "pr_baseline_test.png",
            ),
        }
        log.info(
            f"baseline test: acc={results['baseline']['test']['accuracy']:.4f} "
            f"macro_f1={results['baseline']['test']['macro_f1']:.4f}"
        )
        error_analysis(splits.test, y_true=y_test, y_pred=test_pred).to_csv(
            config.REPORTS_DIR / "errors_baseline.csv", index=False
        )

    # ── Transformer ───────────────────────────────────────────────────────
    if not args.skip_transformer:
        epochs = config.QUICK_TRANSFORMER_EPOCHS if args.quick else config.TRANSFORMER_EPOCHS
        with Timer(f"train transformer (DistilBERT, epochs={epochs})", log):
            model_dir = train_transformer(splits.train, splits.val, epochs=epochs)

        with Timer("evaluate transformer", log):
            val_pred_t, _ = predict_transformer(splits.val, model_dir=model_dir)
            test_pred_t, _ = predict_transformer(splits.test, model_dir=model_dir)
            results["transformer"] = {
                "model_dir": str(model_dir),
                "val": _eval_and_plot(
                    "transformer_val", splits.val, y_val, val_pred_t,
                    cm_path=config.FIGURES_DIR / "cm_transformer_val.png",
                    pr_path=config.FIGURES_DIR / "pr_transformer_val.png",
                ),
                "test": _eval_and_plot(
                    "transformer_test", splits.test, y_test, test_pred_t,
                    cm_path=config.FIGURES_DIR / "cm_transformer_test.png",
                    pr_path=config.FIGURES_DIR / "pr_transformer_test.png",
                ),
            }
            log.info(
                f"transformer test: acc={results['transformer']['test']['accuracy']:.4f} "
                f"macro_f1={results['transformer']['test']['macro_f1']:.4f}"
            )
            error_analysis(splits.test, y_true=y_test, y_pred=test_pred_t).to_csv(
                config.REPORTS_DIR / "errors_transformer.csv", index=False
            )
    else:
        log.info("Skipping transformer training (--skip-transformer).")

    # ── Visualizations on the full dataset ───────────────────────────────
    _save_visualizations(df, log)

    save_json(results, config.REPORTS_DIR / "results.json")
    log.info(f"Results saved → {config.REPORTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
