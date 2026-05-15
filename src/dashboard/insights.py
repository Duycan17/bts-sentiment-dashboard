"""Rule-based business insight cards driven by NSS, volume, and model agreement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.dashboard.data import aspect_nss_table

Severity = Literal["critical", "warning", "good", "info"]


@dataclass(frozen=True)
class InsightCard:
    title: str
    finding: str
    recommendation: str
    severity: Severity
    metric: str = ""


# ─── Thresholds (tunable, kept in one place) ─────────────────────────────────
_PAIN_NSS = -20.0
_PAIN_VOLUME = 500
_STRENGTH_NSS = 60.0
_STRENGTH_VOLUME = 1000
_LOW_CONF_THRESHOLD = 0.5
_LOW_CONF_PCT_FLAG = 10.0
_ASPECT_AGREE_FLAG = 50.0


def _critical_pain_cards(tbl: pd.DataFrame) -> list[InsightCard]:
    pains = tbl[(tbl["nss"] < _PAIN_NSS) & (tbl["rows"] >= _PAIN_VOLUME)].sort_values("nss")
    cards: list[InsightCard] = []
    for _, r in pains.iterrows():
        cards.append(
            InsightCard(
                title=f"Critical pain: {r['aspect']}",
                finding=(
                    f"NSS = {r['nss']:+.1f}% across {int(r['rows'])} reviews — "
                    f"{r['pct_negative']:.0f}% are negative."
                ),
                recommendation=_recommend_for_aspect(str(r["aspect"]), severity="critical"),
                severity="critical",
                metric=f"NSS {r['nss']:+.1f}%",
            )
        )
    return cards


def _strength_cards(tbl: pd.DataFrame) -> list[InsightCard]:
    wins = tbl[(tbl["nss"] >= _STRENGTH_NSS) & (tbl["rows"] >= _STRENGTH_VOLUME)].sort_values("nss", ascending=False)
    cards: list[InsightCard] = []
    for _, r in wins.iterrows():
        cards.append(
            InsightCard(
                title=f"Strength to amplify: {r['aspect']}",
                finding=f"NSS = {r['nss']:+.1f}% across {int(r['rows'])} reviews — riders consistently praise this.",
                recommendation=_recommend_for_aspect(str(r["aspect"]), severity="good"),
                severity="good",
                metric=f"NSS {r['nss']:+.1f}%",
            )
        )
    return cards


def _model_quality_cards(df: pd.DataFrame) -> list[InsightCard]:
    cards: list[InsightCard] = []
    if df.empty:
        return cards

    aspect_agree = (df["aspect"] == df["aspect_pred"]).mean() * 100
    if aspect_agree < _ASPECT_AGREE_FLAG:
        cards.append(
            InsightCard(
                title="Aspect classifier under-recognizes categories",
                finding=(
                    f"Only {aspect_agree:.1f}% of predicted aspects match the labelled aspect. "
                    "The model collapses several aspects into 'Fare & Payment' and 'Crowding & Comfort'."
                ),
                recommendation=(
                    "Retrain the aspect head with class-weighted loss and additional examples for "
                    "Accessibility, Information & Navigation, and Safety. Treat current aspect_pred as advisory only."
                ),
                severity="warning",
                metric=f"Agreement {aspect_agree:.1f}%",
            )
        )

    low_conf = (df["sentiment_confidence"] < _LOW_CONF_THRESHOLD).mean() * 100
    if low_conf > _LOW_CONF_PCT_FLAG:
        cards.append(
            InsightCard(
                title="Sentiment trust threshold needed",
                finding=f"{low_conf:.1f}% of predictions are below {_LOW_CONF_THRESHOLD:.2f} confidence.",
                recommendation=(
                    "Route low-confidence reviews to manual review or treat as Neutral to avoid skewing "
                    "the dashboard with uncertain labels."
                ),
                severity="info",
                metric=f"Low-conf {low_conf:.1f}%",
            )
        )
    return cards


def _recommend_for_aspect(aspect: str, *, severity: Severity) -> str:
    """Aspect-specific recommendations. Falls back to a generic line."""
    table = {
        ("Accessibility", "critical"):
            "Audit lifts, ramps, and platform gaps at the worst-scored stations and publish a 90-day fix plan. "
            "Add a clear 'report a barrier' channel in the BTS app.",
        ("Information & Navigation", "critical"):
            "Standardize bilingual signage, real-time arrival displays, and in-app station maps. "
            "Pilot at the 5 most-complained stations first and measure NSS lift quarterly.",
        ("Safety & Security", "critical"):
            "Boost visible patrols at off-peak hours, add CCTV coverage at low-light platforms, and "
            "publish monthly incident-response stats to rebuild perceived safety.",
        ("Staff & Service", "critical"):
            "Re-train customer-facing staff on de-escalation and English service scripts; add a tip line "
            "for staff complaints with response SLAs.",
        ("Crowding & Comfort", "critical"):
            "Add peak-hour shuttles on the worst-loaded segments and surface live carriage-load on the app.",
        ("Cleanliness", "critical"):
            "Increase platform & carriage cleaning frequency at top-3 negative stations; track via daily QA photos.",
        ("Punctuality & Reliability", "critical"):
            "Publish a public on-time-performance scorecard and tighten dispatch slack at peak hours.",
        ("Fare & Payment System", "good"):
            "Promote Rabbit Card top-up and multi-trip passes in-station — riders already love the value.",
        ("Route Coverage & Connectivity", "good"):
            "Use this strength as the centerpiece of marketing; add interchange wayfinding to MRT to keep riders happy.",
        ("Train Frequency & Waiting Time", "good"):
            "Maintain the current dispatch interval and publicize the on-time stats to defend brand equity.",
        ("Crowding & Comfort", "good"):
            "Sustain the current rolling stock investment and consider exporting the model to peer systems.",
        ("Cleanliness", "good"):
            "Document and standardize the cleaning playbook; share with peer operators as a marketing asset.",
        ("Overall Experience & Convenience", "good"):
            "Use the testimonials in marketing, and protect the experience with strict SLAs as ridership grows.",
    }
    return table.get((aspect, severity), {
        "critical": f"Run a focused improvement sprint on {aspect} and revisit NSS in 90 days.",
        "good": f"Treat {aspect} as a defensible strength and protect it with monitoring.",
        "warning": f"Investigate {aspect} for a structural fix.",
        "info": f"Track {aspect} closely.",
    }[severity])


def generate_insight_cards(
    df: pd.DataFrame,
    *,
    aspect_col: str = "aspect",
    sentiment_col: str = "sentiment",
    include_model_quality: bool = True,
) -> list[InsightCard]:
    if df.empty:
        return []
    tbl = aspect_nss_table(df, aspect_col=aspect_col, sentiment_col=sentiment_col)
    cards = _critical_pain_cards(tbl) + _strength_cards(tbl)
    if include_model_quality:
        cards += _model_quality_cards(df)
    return cards
