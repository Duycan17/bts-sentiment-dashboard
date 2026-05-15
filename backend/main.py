"""FastAPI application factory."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so src.* imports resolve
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.data import router as data_router
from backend.routers.predict import router as predict_router
from backend.routers.chat import router as chat_router
from backend.services import rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_service.build_vector_store()
    yield


app = FastAPI(
    title="BTS ABSA API",
    description="Aspect-based sentiment analysis for Bangkok BTS Skytrain reviews.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
