"""Bangkok BTS — ABSA Strategic Dashboard.

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import classification_report  # noqa: E402

from src.dashboard import charts, explain  # noqa: E402
from src.dashboard.data import (  # noqa: E402
    Filters,
    apply_filters,
    aspect_col_for,
    aspect_nss_table,
    compute_kpis,
    load_dashboard_df,
    sentiment_col_for,
)
from src.dashboard.insights import InsightCard, generate_insight_cards  # noqa: E402
from src.dashboard.theme import SEVERITY_COLORS, configure_plotly  # noqa: E402
from src.dashboard.wordcloud import render_wordcloud_png  # noqa: E402

st.set_page_config(
    page_title="BTS Sentiment — Strategic Dashboard",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)
configure_plotly()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar filters
# ──────────────────────────────────────────────────────────────────────────────

def _sidebar_filters(df: pd.DataFrame) -> Filters:
    st.sidebar.header("Filters")

    label_source = st.sidebar.radio(
        "Label source",
        options=["ground_truth", "predicted"],
        format_func=lambda v: "Ground truth (analyst-labelled)" if v == "ground_truth" else "Predicted (model)",
        help="Switches every chart between the analyst-labelled `aspect`/`sentiment` and the model's `aspect_pred`/`sentiment_pred`.",
    )
    aspect_col = "aspect_pred" if label_source == "predicted" else "aspect"
    sentiment_col = "sentiment_pred" if label_source == "predicted" else "sentiment"

    min_d = df["created_at_date"].min().date()
    max_d = df["created_at_date"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        date_filter = (start, end)
    else:
        date_filter = None

    aspects = st.sidebar.multiselect(
        "Aspects",
        options=sorted(df[aspect_col].unique()),
        default=[],
    )
    sources = st.sidebar.multiselect(
        "Sources",
        options=sorted(df["source"].unique()),
        default=[],
    )
    bts_lines = st.sidebar.multiselect(
        "BTS line",
        options=sorted(df["bts_line"].unique()),
        default=[],
    )
    sentiments = st.sidebar.multiselect(
        "Sentiment",
        options=["Positive", "Neutral", "Negative"],
        default=[],
    )
    min_conf = st.sidebar.slider("Min sentiment confidence", 0.0, 1.0, 0.0, step=0.05)

    if st.sidebar.button("Reset filters", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("FormSubmitter") or k in {"aspects", "sources", "bts_lines", "sentiments"}:
                continue
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data: `DATA_QUALITY_CHECKLIST_filled.csv` (20,782 reviews, 2012–2026, 10 aspects). "
        "Models: TF-IDF + LogReg baseline, fine-tuned DistilBERT."
    )
    return Filters(
        date_range=date_filter,
        aspects=aspects,
        sources=sources,
        bts_lines=bts_lines,
        sentiments=sentiments,
        min_confidence=min_conf,
        label_source=label_source,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shared rendering primitives
# ──────────────────────────────────────────────────────────────────────────────

def _kpi_strip(df: pd.DataFrame, *, baseline: dict, sentiment_col: str) -> None:
    k = compute_kpis(df, sentiment_col=sentiment_col)
    cols = st.columns(6)
    cols[0].metric("Reviews", f"{k['rows']:,}", f"{k['rows'] - baseline['rows']:+,}" if k["rows"] != baseline["rows"] else None)
    cols[1].metric("NSS", f"{k['nss']:+.1f}%", f"{k['nss'] - baseline['nss']:+.1f}" if abs(k["nss"] - baseline["nss"]) > 0.1 else None)
    cols[2].metric("Positive", f"{k['pct_positive']:.1f}%")
    cols[3].metric("Negative", f"{k['pct_negative']:.1f}%")
    cols[4].metric("Aspects", k["n_aspects"])
    span = "—"
    if k["rows"]:
        span = f"{k['date_min'].date()} → {k['date_max'].date()}"
    cols[5].metric("Date span", span)


def _render_card(card: InsightCard) -> None:
    color = SEVERITY_COLORS.get(card.severity, SEVERITY_COLORS["info"])
    badge = {"critical": "🔴 Critical", "warning": "🟠 Warning", "good": "🟢 Strength", "info": "🔵 Note"}[card.severity]
    with st.container(border=True):
        st.markdown(
            f"<div style='border-left:6px solid {color}; padding-left:12px;'>"
            f"<div style='color:{color}; font-weight:600; font-size:0.9rem;'>{badge} · {card.metric}</div>"
            f"<div style='font-size:1.1rem; font-weight:600; margin-top:2px;'>{card.title}</div>"
            f"<div style='margin-top:6px;'><b>Finding:</b> {card.finding}</div>"
            f"<div style='margin-top:4px;'><b>Recommendation:</b> {card.recommendation}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame, baseline_kpis: dict, sentiment_col: str) -> None:
    st.title("Bangkok BTS — Strategic Sentiment Dashboard")
    st.caption(
        "Aspect-based sentiment analysis on 20,782 rider reviews. "
        "Use the sidebar to slice the data; every page reflects the same filter set."
    )

    _kpi_strip(df, baseline=baseline_kpis, sentiment_col=sentiment_col)

    st.markdown("### Top recommendations")
    cards = generate_insight_cards(df, sentiment_col=sentiment_col)
    if not cards:
        st.info("No data after filters. Loosen them to surface insights.")
    else:
        rec_cols = st.columns(2)
        for i, card in enumerate(cards[:6]):
            with rec_cols[i % 2]:
                _render_card(card)

    st.markdown("### Composition")
    a, b = st.columns([1, 1])
    a.plotly_chart(charts.sentiment_donut(df, sentiment_col=sentiment_col), use_container_width=True)
    b.plotly_chart(charts.rating_sentiment_heatmap(df, sentiment_col=sentiment_col), use_container_width=True)

    c, d = st.columns([1, 1])
    c.plotly_chart(charts.source_mix(df), use_container_width=True)
    d.plotly_chart(charts.line_mix(df), use_container_width=True)


def page_aspect_pulse(df: pd.DataFrame, aspect_col: str, sentiment_col: str) -> None:
    st.title("Aspect Pulse")
    st.caption("Where the brand wins or loses — sized and ranked by NSS and review volume.")

    st.plotly_chart(charts.nss_by_aspect_bar(df, aspect_col=aspect_col, sentiment_col=sentiment_col), use_container_width=True)

    a, b = st.columns([1, 1])
    a.plotly_chart(charts.aspect_sentiment_heatmap(df, aspect_col=aspect_col, sentiment_col=sentiment_col), use_container_width=True)
    b.plotly_chart(charts.volume_nss_quadrant(df, aspect_col=aspect_col, sentiment_col=sentiment_col), use_container_width=True)

    st.markdown("### Aspect breakdown table")
    tbl = aspect_nss_table(df, aspect_col=aspect_col, sentiment_col=sentiment_col).sort_values("nss")
    st.dataframe(
        tbl[["aspect", "rows", "pos", "neu", "neg", "nss", "pct_negative"]].rename(
            columns={
                "rows": "Reviews", "pos": "Positive", "neu": "Neutral", "neg": "Negative",
                "nss": "NSS (%)", "pct_negative": "% Negative",
            }
        ).style.format({"NSS (%)": "{:+.1f}", "% Negative": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Recommendations driven by this aspect view")
    for card in generate_insight_cards(df, aspect_col=aspect_col, sentiment_col=sentiment_col, include_model_quality=False):
        _render_card(card)


def page_trends(df: pd.DataFrame, sentiment_col: str, aspect_col: str) -> None:
    st.title("Trends Over Time")
    st.caption("How rider sentiment moved across months and seasons.")

    a, b = st.columns(2)
    a.plotly_chart(charts.monthly_volume_line(df), use_container_width=True)
    b.plotly_chart(charts.monthly_nss_line(df, sentiment_col=sentiment_col), use_container_width=True)

    st.plotly_chart(charts.nss_by_aspect_smallmult(df, aspect_col=aspect_col, sentiment_col=sentiment_col), use_container_width=True)
    st.plotly_chart(charts.seasonality_heatmap(df), use_container_width=True)


def page_voice(df: pd.DataFrame, aspect_col: str, sentiment_col: str) -> None:
    st.title("Voice of Customer")
    st.caption("What riders are actually saying — by polarity and aspect.")

    aspect_choice = st.selectbox(
        "Filter to a single aspect (optional)",
        options=["(All aspects)"] + sorted(df[aspect_col].unique()),
    )
    work = df if aspect_choice == "(All aspects)" else df[df[aspect_col] == aspect_choice]
    if work.empty:
        st.info("No reviews match this slice.")
        return

    pos_texts = tuple(work.loc[work[sentiment_col] == "Positive", "review_text"].astype(str).head(2000).tolist())
    neg_texts = tuple(work.loc[work[sentiment_col] == "Negative", "review_text"].astype(str).head(2000).tolist())

    st.markdown("#### Word clouds")
    a, b = st.columns(2)
    with a:
        st.markdown("**Positive reviews**")
        png = render_wordcloud_png(pos_texts, colormap="Greens")
        if png:
            st.image(png, use_container_width=True)
        else:
            st.info("Not enough positive text for a word cloud.")
    with b:
        st.markdown("**Negative reviews**")
        png = render_wordcloud_png(neg_texts, colormap="Reds")
        if png:
            st.image(png, use_container_width=True)
        else:
            st.info("Not enough negative text for a word cloud.")

    st.markdown("#### Top phrases")
    c, d = st.columns(2)
    c.plotly_chart(
        charts.top_ngrams_bar(list(pos_texts), n=20, ngram=(2, 2), title="Top bigrams — Positive", color="#4CAF50"),
        use_container_width=True,
    )
    d.plotly_chart(
        charts.top_ngrams_bar(list(neg_texts), n=20, ngram=(2, 2), title="Top bigrams — Negative", color="#F44336"),
        use_container_width=True,
    )


def page_performance(df: pd.DataFrame) -> None:
    st.title("Model Performance")
    st.caption("Where the sentiment model can be trusted — and where to be cautious.")

    if df.empty:
        st.info("No data after filters.")
        return

    y_true = df["sentiment"].astype(str).to_numpy()
    y_pred = df["sentiment_pred"].astype(str).to_numpy()

    st.plotly_chart(charts.confusion_matrix_fig(y_true, y_pred, title="Sentiment confusion matrix"), use_container_width=True)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    a, b = st.columns([1, 1])
    a.plotly_chart(charts.per_class_metrics_bar(report), use_container_width=True)
    b.plotly_chart(charts.confidence_calibration(df), use_container_width=True)

    st.markdown("### Per-aspect F1 (weighted)")
    rows = []
    for asp, sub in df.groupby("aspect"):
        if len(sub) < 30:
            continue
        rep = classification_report(sub["sentiment"], sub["sentiment_pred"], output_dict=True, zero_division=0)
        rows.append({
            "Aspect": asp,
            "Reviews": len(sub),
            "Accuracy": rep["accuracy"],
            "Macro F1": rep["macro avg"]["f1-score"],
            "Weighted F1": rep["weighted avg"]["f1-score"],
        })
    if rows:
        perf = pd.DataFrame(rows).sort_values("Weighted F1", ascending=False)
        st.dataframe(
            perf.style.format({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}", "Weighted F1": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )

    st.markdown("### Aspect-prediction agreement (diagnostic)")
    agree = (df["aspect"] == df["aspect_pred"]).mean() * 100
    st.metric("Aspect agreement", f"{agree:.1f}%",
              help="Fraction of rows where the predicted aspect matches the analyst-labelled aspect.")
    st.plotly_chart(charts.aspect_agreement_diagnostic(df), use_container_width=True)


def page_predictor(df: pd.DataFrame) -> None:
    st.title("Live Predictor + SHAP")
    st.caption(
        "Paste a review, get an aspect + sentiment prediction with confidence, and click **Explain** "
        "to see which tokens pushed the model toward the predicted class."
    )

    sample_options = {
        "Strength — fare/payment": "BTS is super convenient and cheap, the rabbit card top-up takes seconds.",
        "Pain — accessibility": "The elevator was broken again and there is no ramp, useless for wheelchair users.",
        "Pain — information & navigation": "The signage is so confusing, I got off at the wrong station because the announcements are too quiet.",
        "Mixed — crowding": "Trains run on time but the carriages are unbearably packed at rush hour.",
    }
    a, b = st.columns([1, 1])
    sample_key = a.selectbox("Or load a sample review", options=["(custom)"] + list(sample_options.keys()))
    aspect_options = sorted(df["aspect"].unique())
    initial_text = sample_options.get(sample_key, "")
    text = st.text_area("Review text", value=initial_text, height=140, key="predictor_text")
    aspect = st.selectbox("Aspect (informs the model's input format)", options=aspect_options, key="predictor_aspect")

    if not text.strip():
        st.info("Type or paste a review to predict.")
        return

    predict_btn = st.button("Predict", type="primary", use_container_width=False)
    explain_btn = st.button("Predict + Explain (SHAP)", type="secondary", use_container_width=False)

    if predict_btn or explain_btn:
        try:
            label, proba = explain.predict_baseline(text, aspect)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted sentiment", label)
        c2.metric("Confidence", f"{proba[label]:.1%}")
        c3.metric("Aspect", aspect)
        st.markdown("**Class probabilities**")
        prob_df = pd.DataFrame({"Class": list(proba.keys()), "Probability": list(proba.values())})
        st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True, hide_index=True)

        if explain_btn:
            with st.spinner("Computing SHAP contributions..."):
                try:
                    result = explain.explain_baseline(text, aspect)
                except Exception as exc:
                    st.error(f"Explanation failed: {exc}")
                    return
            st.plotly_chart(explain.render_token_contributions(result), use_container_width=True)
            st.caption(
                "Bars show each token's signed contribution to the predicted class log-odds. "
                "Coloured bars push toward the predicted class; grey bars push away from it. "
                f"Base log-odds = {result.base_value:+.3f}."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    df_all = load_dashboard_df()
    filters = _sidebar_filters(df_all)
    aspect_col = aspect_col_for(filters)
    sentiment_col = sentiment_col_for(filters)

    df = apply_filters(df_all, filters)

    baseline_kpis = compute_kpis(df_all, sentiment_col=sentiment_col)

    page = st.sidebar.radio(
        "Page",
        options=[
            "Overview",
            "Aspect Pulse",
            "Trends Over Time",
            "Voice of Customer",
            "Model Performance",
            "Live Predictor + SHAP",
        ],
        index=0,
    )

    if df.empty and page != "Live Predictor + SHAP":
        st.warning("No reviews match the current filters. Loosen them in the sidebar.")
        return

    if page == "Overview":
        page_overview(df, baseline_kpis, sentiment_col)
    elif page == "Aspect Pulse":
        page_aspect_pulse(df, aspect_col=aspect_col, sentiment_col=sentiment_col)
    elif page == "Trends Over Time":
        page_trends(df, sentiment_col=sentiment_col, aspect_col=aspect_col)
    elif page == "Voice of Customer":
        page_voice(df, aspect_col=aspect_col, sentiment_col=sentiment_col)
    elif page == "Model Performance":
        page_performance(df)
    elif page == "Live Predictor + SHAP":
        page_predictor(df_all)


if __name__ == "__main__":
    main()
