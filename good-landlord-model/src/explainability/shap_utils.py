"""
shap_utils.py — Global and local SHAP explainability for landlord models.

Public API
----------
prepare_model_frame(matrix, numeric_cols, categorical_cols, model_type)
    -> pd.DataFrame ready for predict / SHAP
compute_shap_explanation(model, X, model_type, max_samples, seed)
    -> shap.Explanation
plot_global_importance(explanation, out_path, top_n)
plot_beeswarm(explanation, out_path, max_display)
plot_dependence(explanation, feature, out_path)
local_top_drivers(explanation, row_index, top_k)
select_case_study_ids(scores_df, n_good, n_bad, n_mid)
build_case_studies(...)
write_explainability_report(...)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.explainability.feature_glossary import (
    driver_phrase,
    enrich_drivers,
    feature_label,
    glossary_as_records,
)

logger = logging.getLogger(__name__)

ModelType = Literal["catboost", "xgboost", "lightgbm", "random_forest"]


# ---------------------------------------------------------------------------
# Feature frame preparation (must match training)
# ---------------------------------------------------------------------------


def prepare_model_frame(
    matrix: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    model_type: str = "catboost",
) -> pd.DataFrame:
    """
    Build the feature frame in the same order/format used at training time.

    CatBoost keeps categoricals as strings (NaN → '__missing__').
    Tree models that used ordinal encoding still accept a DataFrame here for
    SHAP TreeExplainer when the fitted model was a CatBoostRegressor; for
    sklearn/xgb/lgb joblib models that saw numpy arrays, callers should pass
    the encoded array separately (see compute_shap_explanation).
    """
    feat_cols = list(numeric_cols) + list(categorical_cols)
    missing = [c for c in feat_cols if c not in matrix.columns]
    if missing:
        raise ValueError(f"Feature columns missing from matrix: {missing}")

    X = matrix[feat_cols].copy()
    for col in categorical_cols:
        X[col] = X[col].fillna("__missing__").astype(str)
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------


def compute_shap_explanation(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "catboost",
    *,
    max_samples: int = 2000,
    seed: int = 42,
    force_index: list | pd.Index | None = None,
) -> "Any":
    """
    Compute a shap.Explanation using TreeExplainer.

    Samples up to ``max_samples`` rows for speed (global plots).
    ``force_index`` matrix indices are always included (for case studies).
    """
    import shap

    rng = np.random.default_rng(seed)
    n = len(X)
    positions = np.arange(n)

    force_pos: list[int] = []
    if force_index is not None:
        force_set = set(force_index)
        force_pos = [i for i, idx in enumerate(X.index) if idx in force_set]

    if n > max_samples:
        remaining = max_samples - len(force_pos)
        pool = np.setdiff1d(positions, np.array(force_pos, dtype=int), assume_unique=False)
        if remaining > 0 and len(pool) > 0:
            chosen = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            idx = np.sort(np.unique(np.concatenate([np.array(force_pos, dtype=int), chosen])))
        else:
            idx = np.sort(np.array(force_pos, dtype=int)) if force_pos else rng.choice(n, size=max_samples, replace=False)
        X_s = X.iloc[idx].copy()
        sample_index = X.index[idx]
    else:
        X_s = X
        sample_index = X.index

    logger.info(
        "Computing SHAP (%s) on %d / %d rows (forced=%d) …",
        model_type, len(X_s), n, len(force_pos),
    )

    if model_type == "catboost":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_s)
    else:
        X_np = X_s.to_numpy(dtype=float, copy=True)
        X_np = np.nan_to_num(X_np, nan=0.0)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_np)
        try:
            shap_values.feature_names = list(X_s.columns)
        except Exception:
            pass

    shap_values.landlord_index = sample_index  # type: ignore[attr-defined]
    return shap_values


# ---------------------------------------------------------------------------
# Global plots
# ---------------------------------------------------------------------------


def _get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_global_importance(
    explanation: Any,
    out_path: Path | str,
    *,
    top_n: int = 15,
    title: str = "SHAP global feature importance",
) -> Path:
    """Mean |SHAP| bar chart."""
    import shap

    plt = _get_plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    shap.plots.bar(explanation, max_display=top_n, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", out_path)
    return out_path


def plot_beeswarm(
    explanation: Any,
    out_path: Path | str,
    *,
    max_display: int = 15,
    title: str = "SHAP beeswarm (feature effects)",
) -> Path:
    """Beeswarm / summary plot of SHAP values."""
    import shap

    plt = _get_plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", out_path)
    return out_path


def plot_dependence(
    explanation: Any,
    feature: str | int,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """SHAP dependence plot for one feature."""
    import shap

    plt = _get_plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    shap.plots.scatter(explanation[:, feature], show=False)
    plt.title(title or f"SHAP dependence: {feature}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", out_path)
    return out_path


def plot_local_waterfall(
    explanation: Any,
    row_pos: int,
    out_path: Path | str,
    *,
    title: str | None = None,
    max_display: int = 12,
) -> Path:
    """Local waterfall plot for one explanation row (positional index)."""
    import shap

    plt = _get_plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation[row_pos], max_display=max_display, show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Local drivers + case studies
# ---------------------------------------------------------------------------


def local_top_drivers(
    explanation: Any,
    row_pos: int,
    *,
    top_k: int = 5,
) -> tuple[list[dict], list[dict]]:
    """
    Return (positive_drivers, negative_drivers) for one row.

    Each driver includes raw fields plus human-readable enrichment:
      feature, shap_value, feature_value,
      label, description, value_display, direction, explanation
    """
    values = np.asarray(explanation.values[row_pos], dtype=float)
    names = list(explanation.feature_names) if explanation.feature_names is not None else [
        f"f{i}" for i in range(len(values))
    ]
    data_row = explanation.data[row_pos] if explanation.data is not None else [None] * len(values)

    rows = []
    for name, sv, fv in zip(names, values, data_row):
        rows.append({"feature": name, "shap_value": float(sv), "feature_value": fv})

    pos = sorted([r for r in rows if r["shap_value"] > 0], key=lambda r: -r["shap_value"])[:top_k]
    neg = sorted([r for r in rows if r["shap_value"] < 0], key=lambda r: r["shap_value"])[:top_k]
    return enrich_drivers(pos), enrich_drivers(neg)


def select_case_study_ids(
    scores: pd.DataFrame,
    *,
    score_col: str = "AdjustedScore",
    id_col: str = "LandLordID",
    n_good: int = 2,
    n_bad: int = 2,
    n_mid: int = 1,
    min_tenants: int = 5,
    tenant_col: str = "TenantCount",
) -> pd.DataFrame:
    """
    Pick representative landlords for case studies.

    Prefers landlords with enough tenants for stable scores.
    Returns a DataFrame with columns: LandLordID, band, AdjustedScore, TenantCount, …
    """
    df = scores.copy()
    if tenant_col in df.columns:
        eligible = df[df[tenant_col] >= min_tenants].copy()
        if len(eligible) < (n_good + n_bad + n_mid):
            eligible = df.copy()
    else:
        eligible = df.copy()

    eligible = eligible.sort_values(score_col, ascending=False).reset_index(drop=True)
    cases = []

    for _, row in eligible.head(n_good).iterrows():
        cases.append({**row.to_dict(), "band": "good"})

    for _, row in eligible.tail(n_bad).iterrows():
        cases.append({**row.to_dict(), "band": "bad"})

    mid_start = max(0, len(eligible) // 2 - n_mid // 2)
    for _, row in eligible.iloc[mid_start : mid_start + n_mid].iterrows():
        cases.append({**row.to_dict(), "band": "neutral"})

    out = pd.DataFrame(cases)
    # Deduplicate if overlap
    out = out.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    return out


def _fmt_val(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    return str(v)


def _narrative(
    band: str,
    landlord_id: str,
    score: float,
    pred: float | None,
    tenant_count: int | None,
    pos: list[dict],
    neg: list[dict],
) -> str:
    pos_txt = "; ".join(driver_phrase(d) for d in pos[:3]) or "no strong upward drivers"
    neg_txt = "; ".join(driver_phrase(d) for d in neg[:3]) or "no strong downward drivers"
    conf = (
        f"based on {tenant_count} matched tenants"
        if tenant_count is not None
        else "tenant count unavailable"
    )
    pred_txt = (
        f", and the model predicted {pred:.4f}"
        if pred is not None and not np.isnan(pred)
        else ""
    )
    return (
        f"Landlord `{landlord_id}` is a **{band}** case with historical AdjustedScore "
        f"{score:.4f}{pred_txt} ({conf}). "
        f"What pushed the score up: {pos_txt}. "
        f"What pulled the score down: {neg_txt}. "
        f"These are model associations used to explain the prediction — "
        f"not proof that the landlord caused those tenant outcomes."
    )


def build_case_studies(
    matrix: pd.DataFrame,
    explanation: Any,
    case_rows: pd.DataFrame,
    *,
    id_col: str = "LandLordID",
    score_col: str = "AdjustedScore",
    pred_col: str | None = "oof_pred",
    tenant_col: str = "TenantCount",
    figures_dir: Path | str,
    top_k: int = 5,
) -> list[dict]:
    """
    Build structured case studies with local SHAP drivers and waterfall plots.

    ``explanation`` must be computed on a frame whose index aligns with
    ``matrix.index`` (or explanation.landlord_index).
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Map landlord id → positional index inside explanation sample
    if hasattr(explanation, "landlord_index"):
        sample_index = list(explanation.landlord_index)
    else:
        sample_index = list(range(len(explanation.values)))

    # Build lookup from LandLordID → explanation row position
    id_to_pos: dict[str, int] = {}
    for pos, idx in enumerate(sample_index):
        lid = str(matrix.loc[idx, id_col]) if idx in matrix.index else None
        if lid is not None:
            id_to_pos[lid] = pos

    # If case landlord wasn't in SHAP sample, we still describe from full matrix
    # but skip waterfall unless we can find them.
    studies: list[dict] = []
    for _, case in case_rows.iterrows():
        lid = str(case[id_col])
        band = str(case.get("band", "unknown"))
        score = float(case[score_col])
        tenants = int(case[tenant_col]) if tenant_col in case and pd.notna(case[tenant_col]) else None
        pred = float(case[pred_col]) if pred_col and pred_col in case and pd.notna(case[pred_col]) else None

        waterfall_path = None
        pos_drivers: list[dict] = []
        neg_drivers: list[dict] = []

        if lid in id_to_pos:
            row_pos = id_to_pos[lid]
            pos_drivers, neg_drivers = local_top_drivers(explanation, row_pos, top_k=top_k)
            safe_id = lid.replace("/", "_").replace("\\", "_")
            waterfall_path = str(
                plot_local_waterfall(
                    explanation,
                    row_pos,
                    figures_dir / f"waterfall_{band}_{safe_id}.png",
                    title=f"{band.upper()} case: {lid}",
                )
            )
        else:
            logger.warning(
                "Landlord %s not in SHAP sample — case study without waterfall.", lid
            )

        studies.append({
            "LandLordID": lid,
            "band": band,
            "AdjustedScore": score,
            "oof_pred": pred,
            "TenantCount": tenants,
            "top_positive_drivers": pos_drivers,
            "top_negative_drivers": neg_drivers,
            "waterfall_path": waterfall_path,
            "narrative": _narrative(band, lid, score, pred, tenants, pos_drivers, neg_drivers),
        })

    return studies


def mean_abs_shap_table(explanation: Any, top_n: int = 20) -> pd.DataFrame:
    """Return DataFrame of mean |SHAP| by feature with human labels."""
    values = np.abs(np.asarray(explanation.values, dtype=float)).mean(axis=0)
    names = list(explanation.feature_names) if explanation.feature_names is not None else [
        f"f{i}" for i in range(len(values))
    ]
    df = pd.DataFrame({
        "feature": names,
        "label": [feature_label(n) for n in names],
        "mean_abs_shap": values,
    })
    return df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def _rel_for_md(path: Path | str, report_dir: Path) -> str:
    """Path relative to the report file, forward-slash style for Markdown."""
    p = Path(path)
    try:
        return p.resolve().relative_to(report_dir.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def write_explainability_report(
    *,
    model_type: str,
    global_importance: pd.DataFrame,
    case_studies: list[dict],
    figure_paths: dict[str, str],
    out_path: Path | str,
    n_shap_rows: int,
    n_landlords: int,
) -> Path:
    """Write Markdown explainability + case-study report."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = out_path.parent

    lines: list[str] = []
    lines.append("# SHAP Explainability Report")
    lines.append("")
    lines.append(f"_Generated at: {pd.Timestamp.utcnow().isoformat()}_")
    lines.append("")
    lines.append(f"- Model: **`{model_type}`**")
    lines.append(f"- Landlords in matrix: **{n_landlords}**")
    lines.append(f"- SHAP sample size: **{n_shap_rows}**")
    lines.append("")
    lines.append(
        "> SHAP explains how the model produced its prediction in plain language. "
        "It does **not** prove that a feature caused company or landlord success."
    )
    lines.append("")

    lines.append("## Feature glossary")
    lines.append("")
    lines.append(
        "Technical column names are mapped to human-readable meanings below "
        "(also applied in case-study narratives)."
    )
    lines.append("")
    lines.append("| Column | Meaning | What it measures |")
    lines.append("| --- | --- | --- |")
    for row in glossary_as_records():
        lines.append(
            f"| `{row['feature']}` | {row['label']} | {row['description']} |"
        )
    lines.append("")

    lines.append("## Global feature importance (mean |SHAP|)")
    lines.append("")
    if not global_importance.empty:
        lines.append("| Meaning | Column | mean |SHAP| |")
        lines.append("| --- | --- | ---: |")
        for _, row in global_importance.iterrows():
            label = row["label"] if "label" in row.index else feature_label(str(row["feature"]))
            lines.append(
                f"| {label} | `{row['feature']}` | {row['mean_abs_shap']:.5f} |"
            )
        lines.append("")
    else:
        lines.append("_No importance rows._")
        lines.append("")

    if figure_paths.get("importance"):
        lines.append(
            f"![Global importance]({_rel_for_md(figure_paths['importance'], report_dir)})"
        )
        lines.append("")
    if figure_paths.get("beeswarm"):
        lines.append("## Beeswarm (direction of effects)")
        lines.append("")
        lines.append(f"![Beeswarm]({_rel_for_md(figure_paths['beeswarm'], report_dir)})")
        lines.append("")

    deps = [k for k in figure_paths if k.startswith("dependence_")]
    if deps:
        lines.append("## Dependence plots")
        lines.append("")
        for k in deps:
            feat = k.replace("dependence_", "")
            lines.append(f"### {feature_label(feat)}")
            lines.append("")
            lines.append(f"_Column: `{feat}`_")
            lines.append("")
            lines.append(f"![{feat}]({_rel_for_md(figure_paths[k], report_dir)})")
            lines.append("")

    lines.append("## Case studies")
    lines.append("")
    for i, cs in enumerate(case_studies, 1):
        lines.append(f"### {i}. [{cs['band'].upper()}] `{cs['LandLordID']}`")
        lines.append("")
        lines.append(
            f"- AdjustedScore: **{cs['AdjustedScore']:.4f}**"
            + (f" · OOF pred: **{cs['oof_pred']:.4f}**" if cs.get("oof_pred") is not None else "")
            + (f" · Tenants: **{cs['TenantCount']}**" if cs.get("TenantCount") is not None else "")
        )
        lines.append("")
        lines.append(cs["narrative"])
        lines.append("")
        if cs.get("top_positive_drivers"):
            lines.append("**What raised the score**")
            lines.append("")
            lines.append("| Meaning | Value | Effect (SHAP) | Explanation |")
            lines.append("| --- | --- | ---: | --- |")
            for d in cs["top_positive_drivers"]:
                label = d.get("label") or feature_label(d["feature"])
                val = d.get("value_display") or _fmt_val(d.get("feature_value"))
                expl = d.get("explanation") or driver_phrase(d)
                lines.append(
                    f"| {label} | {val} | +{d['shap_value']:.4f} | {expl} |"
                )
            lines.append("")
        if cs.get("top_negative_drivers"):
            lines.append("**What lowered the score**")
            lines.append("")
            lines.append("| Meaning | Value | Effect (SHAP) | Explanation |")
            lines.append("| --- | --- | ---: | --- |")
            for d in cs["top_negative_drivers"]:
                label = d.get("label") or feature_label(d["feature"])
                val = d.get("value_display") or _fmt_val(d.get("feature_value"))
                expl = d.get("explanation") or driver_phrase(d)
                lines.append(
                    f"| {label} | {val} | {d['shap_value']:.4f} | {expl} |"
                )
            lines.append("")
        if cs.get("waterfall_path"):
            lines.append(
                f"![Waterfall {cs['LandLordID']}]"
                f"({_rel_for_md(cs['waterfall_path'], report_dir)})"
            )
            lines.append("")

    lines.append("---")
    lines.append(
        "_Use wording such as “associated with” or “contributed to the prediction”; "
        "avoid causal claims unless the study design supports them._"
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote explainability report to %s", out_path)
    return out_path
