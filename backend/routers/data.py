"""Data router — all read-only analytics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from backend.models import (
    AspectDetailResponse,
    AspectNSSResponse,
    AspectReviewsResponse,
    BusinessResponse,
    InsightsResponse,
    KPIsResponse,
    MetaResponse,
    PerformanceResponse,
    TrendResponse,
    VoiceResponse,
)
from backend.services import data_service, business_service, error_service
from backend.services import recommendation_service

router = APIRouter(tags=["data"])


def _params(
    date_start: str | None = Query(None),
    date_end: str | None = Query(None),
    aspects: list[str] = Query(default=[]),
    sources: list[str] = Query(default=[]),
    bts_lines: list[str] = Query(default=[]),
    sentiments: list[str] = Query(default=[]),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    label_source: str = Query(default="ground_truth"),
):
    class P:
        pass
    p = P()
    p.date_start = date_start
    p.date_end = date_end
    p.aspects = aspects
    p.sources = sources
    p.bts_lines = bts_lines
    p.sentiments = sentiments
    p.min_confidence = min_confidence
    p.label_source = label_source
    return p


@router.get("/meta", response_model=MetaResponse)
def meta():
    return data_service.get_meta()


@router.get("/kpis", response_model=KPIsResponse)
def kpis(p=Depends(_params)):
    return data_service.get_kpis(p)


@router.get("/aspects", response_model=AspectNSSResponse)
def aspects(p=Depends(_params)):
    return data_service.get_aspects(p)


@router.get("/trends", response_model=TrendResponse)
def trends(p=Depends(_params)):
    return data_service.get_trends(p)


@router.get("/voice", response_model=VoiceResponse)
def voice(p=Depends(_params)):
    return data_service.get_voice(p)


@router.get("/wordcloud/{polarity}")
def wordcloud(polarity: str, p=Depends(_params)):
    png = data_service.get_wordcloud_png(polarity, p)
    if not png:
        return Response(content=b"", media_type="image/png")
    return Response(content=png, media_type="image/png")


@router.get("/performance", response_model=PerformanceResponse)
def performance(p=Depends(_params)):
    return data_service.get_performance(p)


@router.get("/insights", response_model=InsightsResponse)
def insights(p=Depends(_params)):
    return data_service.get_insights(p)


@router.get("/business", response_model=BusinessResponse)
def business(p=Depends(_params)):
    return business_service.get_business(p)


@router.get("/error-analysis")
def error_analysis(p=Depends(_params)):
    return error_service.get_error_analysis(p)
def aspect_reviews(aspect: str, p=Depends(_params)):
    return data_service.get_aspect_reviews(aspect, p)


@router.get("/aspect-detail/{aspect}", response_model=AspectDetailResponse)
def aspect_detail(aspect: str, p=Depends(_params)):
    return data_service.get_aspect_detail(aspect, p)


@router.get("/aspect-recommendation/{aspect}")
def aspect_recommendation(aspect: str):
    return recommendation_service.get_recommendation(aspect)
