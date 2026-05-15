FROM python:3.11-slim

WORKDIR /app

# System deps for wordcloud + Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi "uvicorn[standard]" \
    pandas numpy scikit-learn joblib \
    shap wordcloud Pillow pyarrow \
    imbalanced-learn plotly seaborn matplotlib \
    openai chromadb

# Copy source
COPY src/ ./src/
COPY backend/ ./backend/
COPY DATA_QUALITY_CHECKLIST_filled.csv .
COPY artifacts/models/baseline.joblib ./artifacts/models/baseline.joblib

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
