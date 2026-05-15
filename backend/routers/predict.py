"""Predict + explain endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.models import ExplainResponse, PredictRequest, PredictResponse
from backend.services import explain_service

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return explain_service.predict(req.text, req.aspect)


@router.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest):
    return explain_service.explain(req.text, req.aspect)
