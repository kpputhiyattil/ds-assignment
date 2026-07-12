"""Streamlit UI: upload invoices → dual-model credit assessment via FastAPI.

Shows, for each customer:

* inferred billing interval + human reason
* decision WITH inferred-interval features + human reason
* decision WITHOUT inferred-interval features + human reason
* side-by-side comparison narrative

Run (API must already be up)::

    uvicorn src.serving.api:app --port 8000
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import datetime as dt
import json

import httpx
import pandas as pd
import streamlit as st

DEFAULT_API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Billing Interval · Credit Assessment",
    page_icon="📋",
    layout="wide",
)

st.title("Billing Interval → Credit Assessment")
st.caption(
    "Upload a CSV/Parquet invoice file. Every customer is scored by **two** models "
    "(with vs without the inferred billing interval) through FastAPI, with "
    "human-readable reasons for the interval and each recommendation."
)

with st.sidebar:
    st.header("API")
    api_url = st.text_input("Base URL", value=DEFAULT_API).rstrip("/")
    try:
        _info = httpx.get(f"{api_url}/model/info", timeout=5.0)
        if _info.status_code == 200:
            _info_j = _info.json()
            if _info_j.get("dual_models"):
                st.success(
                    f"Dual package ready · v{_info_j.get('model_version')} · "
                    f"{_info_j.get('n_features')} feats (with-interval)"
                )
            else:
                st.error(
                    "API is up but **not** dual. Kill every uvicorn on port 8000, "
                    "then start with: `.venv\\Scripts\\python -m uvicorn "
                    "src.serving.api:app --host 127.0.0.1 --port 8000`"
                )
        else:
            st.warning(f"/model/info → {_info.status_code}")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"API unreachable: {exc}")
    if st.button("Check health"):
        try:
            r = httpx.get(f"{api_url}/health", timeout=10.0)
            st.json(r.json())
        except Exception as exc:  # noqa: BLE001
            st.error(f"API unreachable: {exc}")
    if st.button("Model info"):
        try:
            r = httpx.get(f"{api_url}/model/info", timeout=10.0)
            st.json(r.json())
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    if st.button("Reload dual package"):
        try:
            r = httpx.post(f"{api_url}/reload", timeout=120.0)
            if r.status_code == 200:
                st.success(r.json())
                st.rerun()
            else:
                st.error(r.text)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))


def _reco_color(reco: str) -> str:
    return {"Continue": "green", "Review": "orange", "No-Go": "red"}.get(reco, "gray")


def _highlight_card(title: str, reco: str, body_md: str) -> None:
    color = _reco_color(reco)
    st.markdown(
        f"""
<div style="border:2px solid #444; border-left:8px solid;
            border-left-color: {'#2e7d32' if reco=='Continue' else '#ef6c00' if reco=='Review' else '#c62828'};
            padding:0.9rem 1rem; margin-bottom:0.8rem; border-radius:6px;">
  <div style="font-size:0.85rem; opacity:0.75;">{title}</div>
  <div style="font-size:1.25rem; font-weight:700;">Decision: <span style="color:{'#2e7d32' if reco=='Continue' else '#ef6c00' if reco=='Review' else '#c62828'}">{reco}</span></div>
  <div style="margin-top:0.5rem; line-height:1.45;">{body_md}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_customer(row: dict, *, expanded: bool = False) -> None:
    cid = row.get("customer_id")
    with_b = row.get("with_interval") or {}
    without_b = row.get("without_interval") or {}
    cmp_ = row.get("comparison") or {}
    label = (
        f"{cid} · interval={row.get('inferred_interval')} · "
        f"WITH {with_b.get('recommendation')} / WITHOUT {without_b.get('recommendation')}"
    )
    with st.expander(label, expanded=expanded):
        st.markdown("#### 1) Inferred billing interval")
        st.markdown(
            f"**Label:** `{row.get('inferred_interval')}` · "
            f"**Confidence:** {row.get('interval_confidence')}"
        )
        st.info(row.get("interval_reason") or "No interval explanation available.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 2) Model WITH inferred interval")
            _highlight_card(
                "Trained including interval features",
                str(with_b.get("recommendation", "?")),
                (
                    f"P(churn)={with_b.get('churn_probability')} · "
                    f"Expected ${with_b.get('expected_dollar_churn')}<br/><br/>"
                    f"{with_b.get('decision_reason', '')}<br/><br/>"
                    f"<em>{with_b.get('model_drivers', '')}</em>"
                ),
            )
        with c2:
            st.markdown("#### 3) Model WITHOUT inferred interval")
            _highlight_card(
                "Trained on core + gap features only",
                str(without_b.get("recommendation", "?")),
                (
                    f"P(churn)={without_b.get('churn_probability')} · "
                    f"Expected ${without_b.get('expected_dollar_churn')}<br/><br/>"
                    f"{without_b.get('decision_reason', '')}<br/><br/>"
                    f"<em>{without_b.get('model_drivers', '')}</em>"
                ),
            )

        st.markdown("#### Comparison")
        agree = cmp_.get("recommendations_agree")
        st.markdown(
            f"**Agree?** {'Yes' if agree else '**No — decisions differ**'} · "
            f"Δ P(churn)={cmp_.get('churn_probability_delta')} · "
            f"Δ expected $={cmp_.get('expected_dollar_churn_delta')}"
        )
        st.success(cmp_.get("narrative") or "")


def _results_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        w = r.get("with_interval") or {}
        o = r.get("without_interval") or {}
        cmp_ = r.get("comparison") or {}
        rows.append(
            {
                "customer_id": r.get("customer_id"),
                "inferred_interval": r.get("inferred_interval"),
                "interval_confidence": r.get("interval_confidence"),
                "interval_reason": r.get("interval_reason"),
                "with_recommendation": r.get("with_recommendation") or w.get("recommendation"),
                "with_p_churn": r.get("with_p_churn") or w.get("churn_probability"),
                "with_decision_reason": r.get("with_decision_reason")
                or w.get("decision_reason"),
                "without_recommendation": r.get("without_recommendation")
                or o.get("recommendation"),
                "without_p_churn": r.get("without_p_churn") or o.get("churn_probability"),
                "without_decision_reason": r.get("without_decision_reason")
                or o.get("decision_reason"),
                "recommendations_agree": r.get("recommendations_agree")
                if r.get("recommendations_agree") is not None
                else cmp_.get("recommendations_agree"),
                "comparison_narrative": r.get("comparison_narrative")
                or cmp_.get("narrative"),
            }
        )
    return pd.DataFrame(rows)


st.markdown(
    "Upload an invoice extract (`customerid`, `date`, `amount`). "
    "⚠️ Interactive loads are **customer-capped** (default 500) so the full 22M-row "
    "file cannot crash the service."
)
uploaded = st.file_uploader("Invoice file", type=["csv", "parquet", "tsv"])
c1, c2, c3 = st.columns(3)
with c1:
    file_snapshot = st.date_input("Snapshot", value=dt.date(2025, 3, 5), key="file_snap")
with c2:
    explain_top_n = st.number_input(
        "Deep SHAP explain top-N",
        min_value=0,
        max_value=50,
        value=15,
        help="Every row still gets interval + decision reasons; SHAP driver text is expanded for top-N.",
    )
with c3:
    max_customers = st.number_input(
        "Max customers",
        min_value=1,
        max_value=5000,
        value=500,
    )
filter_cid = st.text_input("Optional: score only this customer_id", value="")
recurring_only = st.checkbox("Recurring only (>=2 invoice days)", value=False)

max_mb = 128.0
try:
    info = httpx.get(f"{api_url}/model/info", timeout=5.0).json()
    max_mb = float(info.get("load_limits", {}).get("max_upload_mb", max_mb))
except Exception:  # noqa: BLE001
    pass
st.caption(f"Interactive upload limit: **{max_mb:.0f} MB**.")

if st.button("Assess file", type="primary", key="file_go") and uploaded is not None:
    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    if size_mb > max_mb:
        st.error(
            f"File is {size_mb:.1f} MB; interactive upload limit is {max_mb:.0f} MB."
        )
        st.stop()
    files = {
        "file": (
            uploaded.name,
            uploaded.getvalue(),
            uploaded.type or "application/octet-stream",
        )
    }
    data = {
        "snapshot": file_snapshot.isoformat(),
        "explain_top_n": str(int(explain_top_n)),
        "max_customers": str(int(max_customers)),
        "recurring_only": "true" if recurring_only else "false",
    }
    if filter_cid.strip():
        data["customer_id"] = filter_cid.strip()
    try:
        with st.spinner("Calling FastAPI /assess/file (dual models)…"):
            r = httpx.post(
                f"{api_url}/assess/file",
                files=files,
                data=data,
                timeout=600.0,
            )
        if r.status_code != 200:
            st.error(f"{r.status_code}: {r.text}")
        else:
            body = r.json()
            if body.get("warning"):
                st.warning(body["warning"])

            results = body.get("results") or []
            cmp_sum = body.get("comparison_summary") or {}
            summary = body.get("summary") or {}
            dual_ok = bool(cmp_sum) and bool(
                results
                and (
                    results[0].get("without_interval")
                    or results[0].get("without_recommendation")
                )
            )
            if not dual_ok:
                st.error(
                    "API returned a **single-model** response (no without-interval "
                    "fields). Usually a second/old uvicorn is still bound to port 8000 "
                    "(e.g. system Python instead of `.venv`).\n\n"
                    "1. Stop **all** processes on port 8000\n"
                    "2. Start: `.venv\\Scripts\\python -m uvicorn "
                    "src.serving.api:app --host 127.0.0.1 --port 8000`\n"
                    "3. Sidebar should show **Dual package ready**\n"
                    "4. Re-upload the file"
                )
                with st.expander("Raw API response (debug)"):
                    st.json({
                        "summary_keys": list(summary.keys()),
                        "comparison_summary": cmp_sum,
                        "first_result_keys": list(results[0].keys()) if results else [],
                    })
                st.stop()

            st.subheader("File-level comparison (with vs without interval)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Customers", summary.get("n_customers"))
            m2.metric(
                "Agreement rate",
                cmp_sum.get("agreement_rate", summary.get("agreement_rate")),
            )
            m3.metric(
                "Mean P(churn) WITH",
                cmp_sum.get(
                    "mean_churn_probability_with",
                    summary.get("mean_churn_probability"),
                ),
            )
            m4.metric(
                "Mean P(churn) WITHOUT",
                cmp_sum.get("mean_churn_probability_without"),
            )
            st.info(cmp_sum.get("narrative") or "")

            holdout = cmp_sum.get("offline_holdout_metrics") or {}
            if holdout:
                with st.expander("Offline holdout metrics (from packaging / train fold)"):
                    st.write(holdout.get("narrative"))
                    st.json(holdout)

            c_a, c_b = st.columns(2)
            with c_a:
                st.markdown("**Recommendations WITH interval**")
                st.write(summary.get("recommendation_counts"))
            with c_b:
                st.markdown("**Recommendations WITHOUT interval**")
                st.write(
                    summary.get("recommendation_counts_without_interval")
                    or cmp_sum.get("recommendation_counts_without_interval")
                )

            st.subheader("Per-customer results (both models)")
            table = _results_table(results)
            st.dataframe(table, use_container_width=True, height=360)

            disagree = [
                row
                for row in results
                if not (
                    row.get("recommendations_agree")
                    if row.get("recommendations_agree") is not None
                    else (row.get("comparison") or {}).get("recommendations_agree")
                )
            ]
            st.markdown(
                f"**Disagreements:** {len(disagree)} / {len(results)} "
                "(these are the clearest examples of interval value)"
            )
            show = disagree[:20] if disagree else results[:10]
            for i, row in enumerate(show):
                _render_customer(row, expanded=(i == 0))

            st.download_button(
                "Download full dual-model results JSON",
                data=json.dumps(body, indent=2),
                file_name="dual_assess.json",
                mime="application/json",
            )
            st.download_button(
                "Download comparison CSV",
                data=table.to_csv(index=False),
                file_name="dual_assess.csv",
                mime="text/csv",
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Request failed: {exc}")
