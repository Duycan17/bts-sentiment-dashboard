from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight


Sentiment = Literal["Positive", "Negative", "Neutral"]


CSV_PATH = Path(__file__).resolve().parent / "all_reviews_merged.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


PSEUDO_ASPECTS = {"Irrelevant", "Positive", "Negative", "Neutral"}

# Canonical aspect taxonomy (edit here if you want fewer/more buckets)
CANONICAL_ASPECTS = [
    "Fare & Payment System",
    "Crowding & Comfort",
    "Punctuality & Reliability",
    "Route Coverage & Connectivity",
    "Accessibility",
    "Safety & Security",
    "Signage & Navigation",
    "Staff & Customer Service",
    "Facilities",
    "Cleanliness & Hygiene",
    "Convenience",
    "Data availability",
    "Overall Experience",
]

# Map dataset variants -> canonical aspect label
ASPECT_CANONICAL_MAP = {
    # staff
    "Staff quality": "Staff & Customer Service",
    "Staff & Customer Service": "Staff & Customer Service",
    # facilities/infrastructure
    "Infrastructure & Facilities": "Facilities",
    "Facilities": "Facilities",
    # cleanliness
    "Cleanliness": "Cleanliness & Hygiene",
    "Cleanliness & Hygiene": "Cleanliness & Hygiene",
    # punctuality
    "Punctuality": "Punctuality & Reliability",
    "Punctuality & Reliability": "Punctuality & Reliability",
    # safety
    "Safety": "Safety & Security",
    "Safety & Security": "Safety & Security",
    # misc direct keep
    "Fare & Payment System": "Fare & Payment System",
    "Crowding & Comfort": "Crowding & Comfort",
    "Route Coverage & Connectivity": "Route Coverage & Connectivity",
    "Accessibility": "Accessibility",
    "Signage & Navigation": "Signage & Navigation",
    "Convenience": "Convenience",
    "Data availability": "Data availability",
    "Overall Experience": "Overall Experience",
    # noisy but salvageable
    "Price fairness": "Fare & Payment System",
}


def _stable_text_id(text: str) -> str:
    normalized = " ".join((text or "").split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def load_raw(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {
        "review_text",
        "review_rating",
        "source",
        "created_at_date",
        "bts_line",
        "reviewer_hometown",
        "aspect",
        "sentiment",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df["aspect_raw"] = df["aspect"].astype(str).str.strip()
    df = df[df["sentiment"].isin(["Positive", "Negative", "Neutral"])].copy()

    df = df[~df["aspect_raw"].isin(PSEUDO_ASPECTS)].copy()

    df["aspect"] = df["aspect_raw"].map(ASPECT_CANONICAL_MAP)
    df = df[df["aspect"].isin(CANONICAL_ASPECTS)].copy()

    df["text_id"] = df["review_text"].astype(str).map(_stable_text_id)
    return df


@dataclass(frozen=True)
class Split:
    train_ids: set[str]
    val_ids: set[str]
    test_ids: set[str]


def make_splits(
    df: pd.DataFrame, *, n_splits: int = 5, random_state: int = 42
) -> Split:
    """
    Leakage-safe split:
    - group by identical normalized text (`text_id`)
    - stratify by sentiment
    """
    if "text_id" not in df.columns:
        raise ValueError("Expected text_id column. Run clean_labels first.")

    y = df["sentiment"].to_numpy()
    groups = df["text_id"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(sgkf.split(np.zeros(len(df)), y, groups))
    # Deterministic fold selection: fold0=test, fold1=val, rest=train
    test_fold = 0
    val_fold = 1 if n_splits > 1 else 0

    test_idx = folds[test_fold][1]
    val_idx = folds[val_fold][1]

    test_ids = set(df.iloc[test_idx]["text_id"].tolist())
    val_ids = set(df.iloc[val_idx]["text_id"].tolist())
    train_ids = set(df["text_id"].tolist()) - test_ids - val_ids

    return Split(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)


def verify_no_group_leakage(df: pd.DataFrame, split: Split) -> None:
    overlap_tv = split.train_ids & split.val_ids
    overlap_tt = split.train_ids & split.test_ids
    overlap_vt = split.val_ids & split.test_ids
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            "Group leakage detected in splits: "
            f"train∩val={len(overlap_tv)} train∩test={len(overlap_tt)} val∩test={len(overlap_vt)}"
        )


def format_pair(aspect: str, text: str) -> str:
    # Single-string format works for TF-IDF and as a fallback for transformers
    return f"Aspect: {aspect}\nReview: {text}"


def build_xy(df: pd.DataFrame, aspect_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = df.apply(lambda r: format_pair(r[aspect_col], r["review_text"]), axis=1).to_numpy()
    y = df["sentiment"].to_numpy()
    return x, y


def train_baseline_tfidf_logreg(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    clf = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=80_000, ngram_range=(1, 2))),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=None,
                    class_weight="balanced",
                    C=2.0,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)
    return clf


def train_baseline_tfidf_logreg_oversampled(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Context7-aligned pattern: oversample only on train split.
    """
    from imblearn.over_sampling import SMOTE

    vec = TfidfVectorizer(max_features=80_000, ngram_range=(1, 2))
    x_vec = vec.fit_transform(x_train)
    # SMOTE requires dense input; this can be memory-heavy.
    x_dense = x_vec.toarray()
    smote = SMOTE(random_state=42, k_neighbors=3)
    x_res, y_res = smote.fit_resample(x_dense, y_train)

    clf = LogisticRegression(max_iter=2500, C=1.5)
    clf.fit(x_res, y_res)

    pipe = Pipeline(steps=[("tfidf", vec), ("logreg", clf)])
    return pipe


def tune_thresholds_ovr(
    clf: Pipeline,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    labels: list[str] = ["Negative", "Neutral", "Positive"],
) -> dict[str, float]:
    """
    Simple one-vs-rest threshold tuning on validation to improve macro-F1.
    """
    proba = clf.predict_proba(x_val)
    label_to_idx = {l: i for i, l in enumerate(clf.classes_)}
    y_true = np.array(y_val)

    thresholds: dict[str, float] = {}
    grid = np.arange(0.15, 0.85, 0.05)
    for label in labels:
        if label not in label_to_idx:
            continue
        idx = label_to_idx[label]
        y_bin = (y_true == label).astype(int)
        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            y_hat_bin = (proba[:, idx] >= t).astype(int)
            f1 = f1_score(y_bin, y_hat_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[label] = best_t
    return thresholds


def predict_with_thresholds(
    clf: Pipeline,
    x: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    proba = clf.predict_proba(x)
    classes = list(clf.classes_)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    default_pred = np.argmax(proba, axis=1)

    preds = []
    for i in range(len(x)):
        candidates = []
        for label, t in thresholds.items():
            idx = class_to_idx.get(label)
            if idx is None:
                continue
            if proba[i, idx] >= t:
                candidates.append((proba[i, idx] - t, label))
        if candidates:
            candidates.sort(reverse=True)
            preds.append(candidates[0][1])
        else:
            preds.append(classes[default_pred[i]])
    return np.array(preds)


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = ["Negative", "Neutral", "Positive"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {"name": name, "labels": labels, "confusion_matrix": cm.tolist(), "report": report}


def evaluate_by_aspect(
    df: pd.DataFrame,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    aspect_col: str,
    min_rows: int = 50,
) -> dict:
    labels = ["Negative", "Neutral", "Positive"]
    out: dict[str, dict] = {}
    idx_pos = {idx: i for i, idx in enumerate(df.index.to_list())}
    for aspect, sub in df.groupby(aspect_col):
        if len(sub) < min_rows:
            continue
        pos = [idx_pos[i] for i in sub.index.to_list()]
        yt = y_true[pos]
        yp = y_pred[pos]
        out[str(aspect)] = {
            "rows": int(len(sub)),
            "report": classification_report(yt, yp, labels=labels, output_dict=True, zero_division=0),
        }
    return out


def train_transformer_sentiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    aspect_col: str,
    model_name: str = "distilbert-base-uncased",
    out_dir: Path = ARTIFACTS_DIR / "sentiment_transformer",
    max_length: int = 256,
    epochs: float = 1.0,
    batch_size: int = 8,
    lr: float = 2e-5,
) -> Path:
    """
    Fine-tune a small transformer for sentiment classification.
    Uses sentence-pair encoding: (aspect, review_text).
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset
    import torch

    label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}
    id2label = {v: k for k, v in label2id.items()}

    def to_hf(df: pd.DataFrame) -> Dataset:
        return Dataset.from_dict(
            {
                "aspect": df[aspect_col].astype(str).tolist(),
                "text": df["review_text"].astype(str).tolist(),
                "labels": [label2id[s] for s in df["sentiment"].astype(str).tolist()],
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(
            batch["aspect"],
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    train_ds = to_hf(train_df).map(tokenize, batched=True, remove_columns=["aspect", "text"])
    val_ds = to_hf(val_df).map(tokenize, batched=True, remove_columns=["aspect", "text"])

    y_train = np.array(train_df["sentiment"].astype(str).map(label2id).tolist())
    classes = np.array([0, 1, 2])
    class_weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        return {"macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1)}

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            cw = class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


def predict_transformer(
    df: pd.DataFrame,
    *,
    aspect_col: str,
    model_dir: Path,
    max_length: int = 256,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    labels = []
    probs_all: list[np.ndarray] = []
    aspects = df[aspect_col].astype(str).tolist()
    texts = df["review_text"].astype(str).tolist()

    id2label = model.config.id2label
    if isinstance(id2label, dict):
        id2label = {int(k): v for k, v in id2label.items()}

    for i in range(0, len(df), batch_size):
        a = aspects[i : i + batch_size]
        t = texts[i : i + batch_size]
        enc = tokenizer(a, t, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = logits.argmax(dim=-1).cpu().numpy().tolist()
        labels.extend([id2label[j] for j in pred_ids])
        probs_all.append(probs)

    probs_np = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, len(id2label)))
    return np.array(labels), probs_np


def tune_thresholds_from_probs(
    probs: np.ndarray,
    y_true: np.ndarray,
    classes: list[str],
    *,
    grid: np.ndarray | None = None,
) -> dict[str, float]:
    if grid is None:
        grid = np.arange(0.15, 0.85, 0.05)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    out: dict[str, float] = {}
    for c in classes:
        idx = class_to_idx[c]
        y_bin = (y_true == c).astype(int)
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            y_hat = (probs[:, idx] >= t).astype(int)
            f1 = f1_score(y_bin, y_hat, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = float(f1), float(t)
        out[c] = best_t
    return out


def predict_labels_with_prob_thresholds(
    probs: np.ndarray, classes: list[str], thresholds: dict[str, float]
) -> np.ndarray:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    argmax_idx = probs.argmax(axis=1)
    pred = []
    for i in range(len(probs)):
        cand = []
        for c, t in thresholds.items():
            idx = class_to_idx[c]
            if probs[i, idx] >= t:
                cand.append((probs[i, idx] - t, c))
        if cand:
            cand.sort(reverse=True)
            pred.append(cand[0][1])
        else:
            pred.append(classes[argmax_idx[i]])
    return np.array(pred)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def infer_aspect_zero_shot(
    texts: Iterable[str],
    *,
    candidate_labels: list[str],
    model_name: str = "typeform/distilbert-base-uncased-mnli",
    batch_size: int = 4,
    max_length: int = 256,
) -> list[str]:
    """
    Pretrained aspect inference only (no training).
    Uses zero-shot classification over the canonical aspect labels.
    """
    from transformers import pipeline

    try:
        import torch

        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1

    clf = pipeline("zero-shot-classification", model=model_name, device=device)

    out: list[str] = []
    batch: list[str] = []
    for t in texts:
        batch.append(str(t))
        if len(batch) >= batch_size:
            preds = clf(batch, candidate_labels=candidate_labels, multi_label=False, truncation=True, max_length=max_length)
            for p in preds:
                out.append(p["labels"][0])
            batch = []

    if batch:
        preds = clf(batch, candidate_labels=candidate_labels, multi_label=False, truncation=True, max_length=max_length)
        for p in preds:
            out.append(p["labels"][0])

    return out


def get_or_create_predicted_aspects(df: pd.DataFrame, *, max_rows: int | None) -> pd.DataFrame:
    """
    Adds `aspect_pred` using a pretrained model and caches to disk.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = ARTIFACTS_DIR / "aspects_pred.parquet"

    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows).copy()

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached = cached.drop_duplicates(subset=["text_id"])
        work = work.merge(cached[["text_id", "aspect_pred"]], on="text_id", how="left")
    else:
        work["aspect_pred"] = None

    missing = work["aspect_pred"].isna()
    if missing.any():
        preds = infer_aspect_zero_shot(
            work.loc[missing, "review_text"].astype(str).tolist(),
            candidate_labels=CANONICAL_ASPECTS,
        )
        work.loc[missing, "aspect_pred"] = preds

        to_cache = work[["text_id", "aspect_pred"]].dropna().drop_duplicates(subset=["text_id"])
        if cache_path.exists():
            prev = pd.read_parquet(cache_path).drop_duplicates(subset=["text_id"])
            to_cache = pd.concat([prev, to_cache], ignore_index=True).drop_duplicates(subset=["text_id"])
        to_cache.to_parquet(cache_path, index=False)

    work["aspect_pred"] = work["aspect_pred"].astype(str)
    return work


def run(args: argparse.Namespace) -> None:
    raw = load_raw(Path(args.csv))
    df = clean_labels(raw)

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    split = make_splits(df, n_splits=5, random_state=42)
    verify_no_group_leakage(df, split)

    df = get_or_create_predicted_aspects(df, max_rows=None)
    aspect_col = "aspect_pred" if args.use_predicted_aspect else "aspect"

    df["split"] = np.where(df["text_id"].isin(split.test_ids), "test", np.where(df["text_id"].isin(split.val_ids), "val", "train"))

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    x_train, y_train = build_xy(train_df, aspect_col=aspect_col)
    x_val, y_val = build_xy(val_df, aspect_col=aspect_col)
    x_test, y_test = build_xy(test_df, aspect_col=aspect_col)

    if len(set(y_train.tolist())) < 2:
        raise ValueError(
            "Train split contains <2 sentiment classes. "
            "Increase dataset size (remove --max_rows) or adjust split settings."
        )

    baseline_balanced = train_baseline_tfidf_logreg(x_train, y_train)
    baseline_oversampled = train_baseline_tfidf_logreg_oversampled(x_train, y_train)

    thresholds_balanced = tune_thresholds_ovr(baseline_balanced, x_val, y_val)
    val_pred_balanced = predict_with_thresholds(baseline_balanced, x_val, thresholds_balanced)
    test_pred_balanced = predict_with_thresholds(baseline_balanced, x_test, thresholds_balanced)
    val_macro_balanced = f1_score(y_val, val_pred_balanced, average="macro", zero_division=0)

    thresholds_over = tune_thresholds_ovr(baseline_oversampled, x_val, y_val)
    val_pred_over = predict_with_thresholds(baseline_oversampled, x_val, thresholds_over)
    test_pred_over = predict_with_thresholds(baseline_oversampled, x_test, thresholds_over)
    val_macro_over = f1_score(y_val, val_pred_over, average="macro", zero_division=0)

    if val_macro_over > val_macro_balanced:
        baseline = baseline_oversampled
        thresholds = thresholds_over
        val_pred = val_pred_over
        test_pred = test_pred_over
        baseline_variant = "tfidf_logreg_smote"
    else:
        baseline = baseline_balanced
        thresholds = thresholds_balanced
        val_pred = val_pred_balanced
        test_pred = test_pred_balanced
        baseline_variant = "tfidf_logreg_classweight_balanced"

    results: dict = {
        "data": {
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "rows_test": int(len(test_df)),
            "aspect_col_used": aspect_col,
        },
        "baseline_tfidf_logreg": {
            "variant_selected": baseline_variant,
            "thresholds": thresholds,
            "val": evaluate("baseline_val", y_val, val_pred),
            "test": evaluate("baseline_test", y_test, test_pred),
            "test_by_aspect": evaluate_by_aspect(test_df, y_true=y_test, y_pred=test_pred, aspect_col=aspect_col),
        },
    }

    if args.train_transformer:
        model_dir = train_transformer_sentiment(
            train_df,
            val_df,
            aspect_col=aspect_col,
            model_name=args.transformer_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        val_pred_t_argmax, val_probs_t = predict_transformer(val_df, aspect_col=aspect_col, model_dir=model_dir)
        test_pred_t_argmax, test_probs_t = predict_transformer(test_df, aspect_col=aspect_col, model_dir=model_dir)
        classes_t = ["Negative", "Neutral", "Positive"]
        thresholds_t = tune_thresholds_from_probs(val_probs_t, y_val, classes_t)
        val_pred_t = predict_labels_with_prob_thresholds(val_probs_t, classes_t, thresholds_t)
        test_pred_t = predict_labels_with_prob_thresholds(test_probs_t, classes_t, thresholds_t)
        results["transformer_sentiment"] = {
            "model_dir": str(model_dir),
            "thresholds": thresholds_t,
            "argmax_val_macro_f1": float(f1_score(y_val, val_pred_t_argmax, average="macro", zero_division=0)),
            "argmax_test_macro_f1": float(f1_score(y_test, test_pred_t_argmax, average="macro", zero_division=0)),
            "val": evaluate("transformer_val", y_val, val_pred_t),
            "test": evaluate("transformer_test", y_test, test_pred_t),
            "test_by_aspect": evaluate_by_aspect(test_df, y_true=y_test, y_pred=test_pred_t, aspect_col=aspect_col),
        }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(results, ARTIFACTS_DIR / "results.json")
    print(json.dumps(results, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=str(CSV_PATH))
    p.add_argument("--max_rows", type=int, default=None, help="Limit rows for quick runs / CPU aspect inference.")
    p.add_argument(
        "--use_predicted_aspect",
        action="store_true",
        help="Use pretrained inferred aspect instead of dataset aspect.",
    )
    p.add_argument("--train_transformer", action="store_true", help="Fine-tune a transformer sentiment model.")
    p.add_argument("--transformer_model", type=str, default="distilbert-base-uncased")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())

