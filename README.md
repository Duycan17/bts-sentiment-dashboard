# BTS Skytrain Sentiment Dashboard

Aspect-based sentiment analysis (ABSA) dashboard for Bangkok BTS Skytrain passenger reviews, with an AI-powered RAG chatbot and per-aspect strategic recommendations.

## Features

- **Dashboard** — KPIs, NSS trends, aspect pulse, voice of customer, model performance
- **Business Insights** — per-aspect deep dives with priority classification
- **AI Chatbot** — floating balloon on every page, answers questions grounded in real reviews
- **Strategic Recommendations** — LLM-generated, cached per aspect, rendered as markdown
- **Live Predictor** — predict sentiment for any review text with SHAP explanations

## Prerequisites

- Python 3.12+
- Node.js 18+
- An OpenAI-compatible LLM API key (see `.env.example`)

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Duycan17/bts-sentiment-dashboard.git
cd bts-sentiment-dashboard
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and fill in your LLM_API_KEY
```

### 3. Set up Python environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 4. Add the data files

The following files are excluded from git (too large). Place them in the project root:

```
all_reviews_cleaned.csv
all_reviews_merged.csv
```

And place the trained sentiment model at:

```
artifacts/sentiment_transformer/model.safetensors
artifacts/sentiment_transformer/config.json
artifacts/sentiment_transformer/tokenizer.json
artifacts/sentiment_transformer/tokenizer_config.json
artifacts/sentiment_transformer/vocab.txt
artifacts/sentiment_transformer/special_tokens_map.json
```

> If you don't have the model, run the training pipeline first (see below).

### 5. Start the backend

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will automatically build the local vector store (ChromaDB) on first startup. This takes ~10 seconds.

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

---

## Training the sentiment model (optional)

If you don't have the pretrained model:

```bash
source .venv/bin/activate
python absa_pipeline.py
```

Or with predicted aspects:

```bash
python absa_pipeline.py --use_predicted_aspect
```

> For CPU-only machines, add `--max_rows 200` to smoke-test end-to-end.

---

## Project Structure

```
.
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── models.py                # Pydantic models
│   ├── routers/
│   │   ├── data.py              # Analytics endpoints
│   │   ├── predict.py           # Prediction endpoints
│   │   └── chat.py              # RAG chat endpoint
│   └── services/
│       ├── data_service.py
│       ├── business_service.py
│       ├── rag_service.py       # ChromaDB + LLM
│       └── recommendation_service.py
├── frontend/
│   └── src/
│       ├── pages/               # Dashboard pages
│       ├── components/          # UI components + ChatBalloon
│       └── api/                 # API client
├── src/
│   └── dashboard/               # Data loading helpers
├── artifacts/
│   ├── aspects_pred.parquet     # Cached aspect predictions
│   └── recommendations/         # Cached AI recommendations (auto-generated)
├── .env.example                 # Environment variable template
├── requirements.txt
└── pyproject.toml
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_API_KEY` | API key for the LLM endpoint | required |
| `LLM_BASE_URL` | Base URL of the OpenAI-compatible API | `https://llm.chiasegpu.vn/v1` |
| `LLM_CHAT_MODEL` | Model name to use | `gpt-5.4` |

---

## Docker (optional)

```bash
# Backend
docker build -f Dockerfile -t bts-backend .
docker run -p 8000:8000 --env-file .env bts-backend

# Frontend
docker build -f Dockerfile.frontend -t bts-frontend .
docker run -p 80:80 bts-frontend
```
