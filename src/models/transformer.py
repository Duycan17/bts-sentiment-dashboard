"""DistilBERT fine-tune for ABSA sentiment with class-weighted loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from src import config


def _to_hf_dataset(df: pd.DataFrame, *, aspect_col: str):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "aspect": df[aspect_col].astype(str).tolist(),
            "text": df["review_text"].astype(str).tolist(),
            "labels": [config.LABEL2ID[s] for s in df["sentiment"].astype(str).tolist()],
        }
    )


def train_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    aspect_col: str = "aspect",
    out_dir: Path = config.TRANSFORMER_DIR,
    model_name: str = config.TRANSFORMER_MODEL,
    epochs: float = config.TRANSFORMER_EPOCHS,
    batch_size: int = config.TRANSFORMER_BATCH,
    lr: float = config.TRANSFORMER_LR,
    max_length: int = config.TRANSFORMER_MAX_LEN,
) -> Path:
    """Fine-tune a transformer with class-weighted CE loss. Returns the saved model dir."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["aspect"], batch["text"], truncation=True, max_length=max_length)

    train_ds = _to_hf_dataset(train_df, aspect_col=aspect_col).map(
        tokenize, batched=True, remove_columns=["aspect", "text"]
    )
    val_ds = _to_hf_dataset(val_df, aspect_col=aspect_col).map(
        tokenize, batched=True, remove_columns=["aspect", "text"]
    )

    y_train = np.array([config.LABEL2ID[s] for s in train_df["sentiment"].astype(str).tolist()])
    classes = np.arange(len(config.SENTIMENT_LABELS))
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(config.SENTIMENT_LABELS),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
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
        save_total_limit=2,
        report_to=[],
        seed=config.RANDOM_SEED,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.TRANSFORMER_PATIENCE)],
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


def predict_transformer(
    df: pd.DataFrame,
    *,
    aspect_col: str = "aspect",
    model_dir: Path = config.TRANSFORMER_DIR,
    batch_size: int = 32,
    max_length: int = config.TRANSFORMER_MAX_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, probs[N, num_classes])."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    aspects = df[aspect_col].astype(str).tolist()
    texts = df["review_text"].astype(str).tolist()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    all_labels: list[str] = []
    all_probs: list[np.ndarray] = []
    for i in range(0, len(df), batch_size):
        a = aspects[i : i + batch_size]
        t = texts[i : i + batch_size]
        enc = tokenizer(a, t, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = logits.argmax(dim=-1).cpu().numpy().tolist()
        all_labels.extend([id2label[j] for j in pred_ids])
        all_probs.append(probs)

    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(id2label)))
    return np.array(all_labels), probs_np
