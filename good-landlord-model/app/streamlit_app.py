"""
Streamlit analyst UI for Landlord Quality Scoring.

Talks to the FastAPI backend (default http://127.0.0.1:8080).
Start with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("LANDLORD_API_URL", "http://127.0.0.1:8080")

st.set_page_config(
    page_title="Landlord Quality Desk",
    page_icon="🏢",
    layout="wide",
)

BAND_COLORS = {"good": "#22c55e", "neutral": "#f59e0b", "bad": "#ef4444"}


def _api(path: str, **kwargs) -> dict | None:
    try:
        r = requests.request(url=f"{API_BASE}{path}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error(
            f"Cannot reach the API at **{API_BASE}**. "
            "Start it first: `python -m uvicorn service.app.main:app --port 8080`"
        )
        return None
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = exc.response.text
        st.error(f"API error ({exc.response.status_code}): {detail}")
        return None


@st.cache_data(ttl=300)
def _health() -> dict | None:
    return _api("/api/health", method="GET")


@st.cache_data(ttl=300)
def _schema() -> dict | None:
    return _api("/api/schema", method="GET")


def _render_band(band: str) -> str:
    color = BAND_COLORS.get(str(band).lower(), "#6b7280")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em">{band}</span>'
    )


def _render_drivers(drivers: list[dict], direction: str) -> None:
    if not drivers:
        return
    for d in drivers:
        shap_val = d.get("shap_value", 0)
        if abs(shap_val) < 0.001:
            continue
        label = d.get("label") or d.get("feature", "?")
        value_display = d.get("value_display", "—")
        missing = d.get("value_missing", False) or value_display == "not in input"
        sign = "+" if shap_val > 0 else ""
        color = "#22c55e" if direction == "up" else "#ef4444"
        val_text = "missing" if missing else f"= {value_display}"
        st.markdown(
            f"<span style='color:{color};font-weight:600'>{sign}{shap_val:.4f}</span> "
            f"**{label}** ({val_text})",
            unsafe_allow_html=True,
        )


# ── Sidebar: health + schema ────────────────────────────────────────────────
with st.sidebar:
    st.title("Landlord Quality Desk")
    health = _health()
    if health and health.get("status") == "ok":
        st.success(f"API ready — {health.get('model_type', '?')} {health.get('model_version', '')}")
    elif health:
        st.warning(f"API status: {health.get('status')}")
    else:
        st.error("API unreachable")
        st.stop()

    schema = _schema()
    if schema:
        with st.expander("Model info"):
            st.json({
                "model_type": schema.get("model_type"),
                "model_version": schema.get("model_version"),
                "pipeline_steps": schema.get("pipeline_steps", []),
                "package_components": schema.get("package_components", []),
                "max_rows": schema.get("max_rows"),
            })
        with st.expander("Feature glossary"):
            glossary = schema.get("feature_glossary", [])
            if glossary:
                st.dataframe(
                    pd.DataFrame(glossary)[["label", "description", "feature"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No glossary available.")

    top_k = st.slider("Top SHAP drivers per landlord", 1, 15, 5)

# ── Main area: two tabs ─────────────────────────────────────────────────────
tab_upload, tab_manual = st.tabs(["Upload file", "Enter details"])

if "scored_payload" not in st.session_state:
    st.session_state.scored_payload = None

with tab_upload:
    st.markdown(
        "Upload a **CSV or Parquet** file with landlord rows. "
        "Raw landlord data is routed through the full FE pipeline; "
        "pre-built feature matrices are scored directly."
    )
    uploaded = st.file_uploader(
        "Choose CSV / Parquet",
        type=["csv", "parquet", "pq"],
        label_visibility="collapsed",
    )
    if uploaded and st.button("Score file", type="primary", key="btn_file"):
        with st.spinner("Scoring with SHAP explanations…"):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            data = {"top_k": str(top_k)}
            st.session_state.scored_payload = _api("/api/score/file", method="POST", files=files, data=data)

with tab_manual:
    st.markdown("Enter a single landlord's features.")
    cols = st.columns(3)
    manual: dict = {}

    with cols[0]:
        manual["LandLordID"] = st.text_input("LandLordID", placeholder="LANDLORD_0001")
        manual["TenantCount"] = st.number_input("TenantCount", min_value=0, value=0, step=1)
        manual["LandlordAge"] = st.number_input("LandlordAge", min_value=0.0, value=0.0, step=1.0)
        manual["Area"] = st.number_input("Area", min_value=0.0, value=0.0, step=100.0)
    with cols[1]:
        manual["TotalPopulationAround"] = st.number_input("TotalPopulationAround", min_value=0.0, value=0.0, step=1000.0)
        manual["ActiveCompanies"] = st.number_input("ActiveCompanies", min_value=0.0, value=0.0, step=1.0)
        manual["PopulationDensity"] = st.number_input("PopulationDensity", min_value=0.0, value=0.0, step=0.1)
        manual["AreaPerCompany"] = st.number_input("AreaPerCompany", min_value=0.0, value=0.0, step=1.0)
    with cols[2]:
        manual["PreferredIndustry"] = st.text_input("PreferredIndustry", placeholder="Technology")
        manual["OriginCity"] = st.text_input("OriginCity", placeholder="Amsterdam")
        manual["OriginCountry"] = st.text_input("OriginCountry", placeholder="Netherlands")

    with st.expander("Portfolio features (optional)"):
        pcols = st.columns(3)
        with pcols[0]:
            manual["PortfolioSize"] = st.number_input("PortfolioSize", min_value=0.0, value=0.0, step=1.0)
            manual["PortfolioActiveRate"] = st.number_input("PortfolioActiveRate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            manual["PortfolioMeanBudget"] = st.number_input("PortfolioMeanBudget", min_value=0.0, value=0.0, step=100.0)
        with pcols[1]:
            manual["PortfolioStdBudget"] = st.number_input("PortfolioStdBudget", min_value=0.0, value=0.0, step=100.0)
            manual["PortfolioMeanSales"] = st.number_input("PortfolioMeanSales", min_value=0.0, value=0.0, step=100.0)
            manual["PortfolioMeanClients"] = st.number_input("PortfolioMeanClients", min_value=0.0, value=0.0, step=1.0)
        with pcols[2]:
            manual["PortfolioMeanRetention"] = st.number_input("PortfolioMeanRetention", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            manual["PortfolioMeanSuccessScore"] = st.number_input("PortfolioMeanSuccessScore", min_value=0.0, value=0.0, step=0.01)
            manual["PortfolioStdSuccessScore"] = st.number_input("PortfolioStdSuccessScore", min_value=0.0, value=0.0, step=0.01)

    if st.button("Score landlord", type="primary", key="btn_manual"):
        record = {k: v for k, v in manual.items() if v not in (None, "", 0, 0.0)}
        if not record:
            st.warning("Enter at least some feature values.")
        else:
            with st.spinner("Scoring…"):
                st.session_state.scored_payload = _api(
                    "/api/score/json",
                    method="POST",
                    json={"landlords": [record], "top_k": top_k},
                )

# ── Results ──────────────────────────────────────────────────────────────────
if st.session_state.scored_payload:
    st.divider()
    payload = st.session_state.scored_payload
    results = payload.get("results", [])

    pipe_note = payload.get("pipeline_note")
    input_warn = payload.get("input_warning")
    if pipe_note:
        st.info(pipe_note)
    if input_warn:
        st.warning(input_warn)

    trunc = payload.get("truncated", False)
    n_in = payload.get("n_input", 0)
    n_out = payload.get("n_scored", 0)
    st.caption(
        f"Scored **{n_out}** of {n_in} row(s)"
        + (f" (truncated to {payload.get('max_rows')})" if trunc else "")
        + f" · {payload.get('model_type', '')} {payload.get('model_version', '')}"
    )

    if not results:
        st.info("No results returned.")
    else:
        summary_df = pd.DataFrame([
            {
                "LandLordID": r.get("LandLordID", "—"),
                "QualityBand": r.get("QualityBand", "—"),
                "PredictedScore": round(r.get("PredictedScore", 0), 4),
                "QualityScore": round(r.get("PredictedQualityScore", 0), 1),
                "Percentile": round(r.get("PercentileRank", 0), 1) if r.get("PercentileRank") is not None else None,
                "Confidence": r.get("ConfidenceLevel", "—"),
            }
            for r in results
        ])

        col_filter, col_band = st.columns([2, 1])
        with col_filter:
            search = st.text_input("Search landlord ID", key="search_id", label_visibility="collapsed", placeholder="Search landlord ID…")
        with col_band:
            band_opt = st.selectbox("Band", ["All", "good", "neutral", "bad"], key="band_filter", label_visibility="collapsed")

        view = summary_df.copy()
        if search:
            view = view[view["LandLordID"].str.contains(search, case=False, na=False)]
        if band_opt != "All":
            view = view[view["QualityBand"].str.lower() == band_opt.lower()]

        st.caption("Click a row to inspect that landlord's score details.")
        event = st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="results_table",
        )

        # Resolve which row was clicked
        selected_rows = event.selection.rows if event and event.selection else []
        if selected_rows:
            sel_idx = selected_rows[0]
            selected_id = view.iloc[sel_idx]["LandLordID"] if sel_idx < len(view) else None
        elif not view.empty:
            selected_id = view.iloc[0]["LandLordID"]
        else:
            selected_id = None

        row = next((r for r in results if r.get("LandLordID") == selected_id), None) if selected_id else None
        if row:
            st.divider()
            band = row.get("QualityBand", "unknown")
            score = row.get("PredictedScore", 0)
            qual = row.get("PredictedQualityScore", 0)
            pct = row.get("PercentileRank")
            conf = row.get("ConfidenceLevel", "—")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Band", band)
            m2.metric("Score", f"{score:.4f}")
            m3.metric("Quality", f"{qual:.1f} / 100")
            m4.metric("Percentile", f"{pct:.1f}" if pct is not None else "—")

            main_reason = row.get("MainReason") or row.get("ExplanationSummary") or ""
            final_interp = row.get("FinalInterpretation", "")
            if main_reason:
                st.markdown(f"**Main reason:** {main_reason}")
            if final_interp:
                st.markdown(f"**Interpretation:** {final_interp}")

            pos_signals = row.get("PositiveSignals", [])
            neg_signals = row.get("NegativeSignals", [])
            if pos_signals or neg_signals:
                sc1, sc2 = st.columns(2)
                with sc1:
                    if pos_signals:
                        st.markdown("**Positive signals**")
                        for s in pos_signals:
                            st.markdown(f"- {s}")
                with sc2:
                    if neg_signals:
                        st.markdown("**Negative signals**")
                        for s in neg_signals:
                            st.markdown(f"- {s}")

            with st.expander("SHAP drivers", expanded=True):
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown("##### Features pushing score up")
                    _render_drivers(row.get("TopPositiveDrivers", []), "up")
                with dc2:
                    st.markdown("##### Features pushing score down")
                    _render_drivers(row.get("TopNegativeDrivers", []), "down")

st.divider()
st.caption("SHAP explains the prediction — not proven causation.")
