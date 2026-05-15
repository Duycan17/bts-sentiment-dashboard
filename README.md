## ABSA pipeline (pretrained aspect + trained sentiment)

This repo contains a practical ABSA workflow for `all_reviews_merged.csv`:

- **Aspect extraction**: pretrained zero-shot classifier (no training)
- **Sentiment**: train models to predict `Positive/Negative/Neutral` from `aspect + review_text`
- **Leakage control**: deduplicate by normalized text hash and use **grouped** splits

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### Run (baseline sentiment model)

- Use dataset aspect (after canonicalization):

```bash
python absa_pipeline.py
```

- Use **pretrained predicted aspect** as the model input (matches the requirement “aspect comes from available model”):

```bash
python absa_pipeline.py --use_predicted_aspect
```

### Notes

- Aspect prediction is cached in `artifacts/aspects_pred.parquet` so reruns are fast.
- For CPU-only machines, start with `--max_rows 200` to smoke-test end-to-end.

