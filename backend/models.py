"""Pydantic response schemas — single source of truth for the API contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# ─── Filter params (shared query-param dataclass) ────────────────────────────

class FilterParams(BaseModel):
    date_start: str | None = None
    date_end: str | None = None
    aspects: list[str] = []
    sources: list[str] = []
    bts_lines: list[str] = []
    sentiments: list[str] = []
    min_confidence: float = 0.0
    label_source: Literal["ground_truth", "predicted"] = "ground_truth"


# ─── Meta ────────────────────────────────────────────────────────────────────

class MetaResponse(BaseModel):
    sources: list[str]
    bts_lines: list[str]
    aspects: list[str]
    date_min: str
    date_max: str


# ─── KPIs ────────────────────────────────────────────────────────────────────

class KPIsResponse(BaseModel):
    rows: int
    nss: float
    pct_positive: float
    pct_neutral: float
    pct_negative: float
    n_aspects: int
    n_sources: int
    date_min: str
    date_max: str
    avg_confidence: float


# ─── Aspects ─────────────────────────────────────────────────────────────────

class AspectRow(BaseModel):
    aspect: str
    rows: int
    pos: int
    neg: int
    neu: int
    nss: float
    pct_negative: float


class AspectNSSResponse(BaseModel):
    aspects: list[AspectRow]


# ─── Trends ──────────────────────────────────────────────────────────────────

class MonthlyRow(BaseModel):
    year_month: str
    rows: int
    nss: float


class SeasonalityCell(BaseModel):
    year: int
    month: int
    rows: int


class TrendResponse(BaseModel):
    monthly: list[MonthlyRow]
    seasonality: list[SeasonalityCell]


# ─── Voice of Customer ───────────────────────────────────────────────────────

class NgramRow(BaseModel):
    ngram: str
    count: int


class VoiceResponse(BaseModel):
    positive_ngrams: list[NgramRow]
    negative_ngrams: list[NgramRow]


# ─── Model Performance ───────────────────────────────────────────────────────

class ConfusionMatrixResponse(BaseModel):
    labels: list[str]
    counts: list[list[int]]
    normalized: list[list[float]]


class ClassMetrics(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class CalibrationBin(BaseModel):
    avg_conf: float
    accuracy: float
    rows: int


class AspectF1Row(BaseModel):
    aspect: str
    rows: int
    accuracy: float
    macro_f1: float
    weighted_f1: float


class PerformanceResponse(BaseModel):
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: ConfusionMatrixResponse
    per_class: list[ClassMetrics]
    calibration: list[CalibrationBin]
    aspect_agreement: float
    aspect_confusion: ConfusionMatrixResponse
    per_aspect_f1: list[AspectF1Row]


# ─── Business Insights (agency deliverable) ──────────────────────────────────

class TrendDirection(BaseModel):
    direction: Literal["improving", "declining", "stable"]
    recent_nss: float   # last 6 months avg
    prior_nss: float    # prior 6 months avg
    delta: float

class AspectDeepDive(BaseModel):
    aspect: str
    nss: float
    rows: int
    pct_negative: float
    pct_positive: float
    top_complaints: list[str]   # top 5 negative bigrams
    top_praises: list[str]      # top 5 positive bigrams
    trend: TrendDirection
    priority: Literal["urgent", "monitor", "leverage"]
    action: str                 # one-line strategic action

class SourceBreakdown(BaseModel):
    source: str
    rows: int
    nss: float
    pct: float  # share of total

class ExecutiveSummary(BaseModel):
    overall_nss: float
    total_reviews: int
    date_range: str
    top_pain: str
    top_strength: str
    trend_direction: Literal["improving", "declining", "stable"]
    headline: str  # one punchy sentence for the C-suite

class BusinessResponse(BaseModel):
    summary: ExecutiveSummary
    aspect_deep_dives: list[AspectDeepDive]
    source_breakdown: list[SourceBreakdown]
    yoy_nss: list[dict]  # [{year, nss, rows}]


# ─── Aspect sample reviews ───────────────────────────────────────────────────

class SampleReview(BaseModel):
    text: str
    rating: int
    source: str
    sentiment: str
    date: str


class AspectReviewsResponse(BaseModel):
    aspect: str
    negative: list[SampleReview]
    positive: list[SampleReview]


class YearlyNSS(BaseModel):
    year: int
    rows: int
    nss: float
    pos: int
    neg: int
    neu: int


class AspectDetailResponse(BaseModel):
    aspect: str
    nss: float
    rows: int
    pct_positive: float
    pct_neutral: float
    pct_negative: float
    priority: str
    trend: TrendDirection
    action: str
    top_complaints: list[str]
    top_praises: list[str]
    yearly_nss: list[YearlyNSS]
    rating_dist: dict[str, int]       # {"1": n, "2": n, ...}
    source_dist: list[SourceBreakdown]
    sample_negative: list[SampleReview]
    sample_positive: list[SampleReview]


class InsightCardResponse(BaseModel):
    title: str
    finding: str
    recommendation: str
    severity: Literal["critical", "warning", "good", "info"]
    metric: str


class InsightsResponse(BaseModel):
    cards: list[InsightCardResponse]


# ─── Predict / Explain ───────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str
    aspect: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    proba: dict[str, float]


class TokenContribution(BaseModel):
    token: str
    value: float


class ExplainResponse(BaseModel):
    label: str
    confidence: float
    proba: dict[str, float]
    contributions: list[TokenContribution]
    base_value: float
