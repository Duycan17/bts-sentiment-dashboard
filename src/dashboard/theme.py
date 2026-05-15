"""Single source of truth for dashboard colors + Plotly defaults."""

from __future__ import annotations

import plotly.io as pio

SENTIMENT_COLORS: dict[str, str] = {
    "Positive": "#4CAF50",
    "Neutral": "#9E9E9E",
    "Negative": "#F44336",
}

SEVERITY_COLORS: dict[str, str] = {
    "critical": "#F44336",
    "warning": "#FF9800",
    "good": "#4CAF50",
    "info": "#3F51B5",
}

PLOTLY_TEMPLATE = "plotly_white"


def configure_plotly() -> None:
    """Pin a single template + font everywhere. Call once at app start."""
    pio.templates.default = PLOTLY_TEMPLATE


def nss_color(value: float) -> str:
    if value < 0:
        return SEVERITY_COLORS["critical"]
    if value < 30:
        return SEVERITY_COLORS["warning"]
    return SEVERITY_COLORS["good"]
