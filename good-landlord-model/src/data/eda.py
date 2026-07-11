"""
eda.py — EDA utilities for the landlord quality scoring project.

Functions
---------
profile_dataframe(df, name)          -> pd.DataFrame   per-column summary
numeric_summary(df)                  -> pd.DataFrame   describe + skew + kurtosis
category_summary(df, cols)           -> pd.DataFrame   value counts + coverage
bridge_diagnostics(bridge)           -> dict            portfolio-size distribution
plot_numeric_distributions(df, ...)  -> None           histogram grid
plot_category_frequencies(df, ...)   -> None           bar-chart grid
plot_portfolio_distribution(bridge)  -> None           companies-per-landlord histogram
plot_correlation_heatmap(df, ...)    -> None           Pearson heatmap for numeric cols

All plot functions return None and call plt.show() unless ax is supplied,
so they work both in notebooks and as standalone scripts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tabular profiling
# ---------------------------------------------------------------------------


def profile_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    """
    Per-column summary: dtype, non-null count, null %, n_unique, top value.

    Parameters
    ----------
    df   : any pandas DataFrame
    name : label used in log output

    Returns
    -------
    pd.DataFrame with one row per column.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        non_null = s.notna().sum()
        null_pct = (n - non_null) / n * 100 if n > 0 else 0.0
        n_unique = s.nunique(dropna=True)

        try:
            top = s.value_counts(dropna=True).index[0] if non_null > 0 else None
        except Exception:
            top = None

        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_null": int(non_null),
                "null_pct": round(null_pct, 2),
                "n_unique": int(n_unique),
                "top_value": str(top) if top is not None else "",
            }
        )

    profile = pd.DataFrame(rows).set_index("column")
    logger.info("Profile of %s (%d rows, %d cols):\n%s", name, n, len(df.columns), profile)
    return profile


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended describe for numeric columns: adds skew and kurtosis rows.

    Returns
    -------
    pd.DataFrame (statistics x columns), numeric columns only.
    """
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        return pd.DataFrame()

    desc = num_df.describe()
    skew_row = pd.DataFrame(
        num_df.skew().rename("skew")
    ).T
    kurt_row = pd.DataFrame(
        num_df.kurtosis().rename("kurtosis")
    ).T
    summary = pd.concat([desc, skew_row, kurt_row])
    return summary.round(4)


def category_summary(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Value counts for categorical / object columns.

    Parameters
    ----------
    df     : DataFrame
    cols   : list of column names; if None, all object/category columns
    top_n  : number of top values per column

    Returns
    -------
    pd.DataFrame with columns [column, value, count, pct].
    """
    if cols is None:
        cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False).head(top_n)
        total = len(df)
        for val, cnt in vc.items():
            rows.append(
                {
                    "column": col,
                    "value": str(val),
                    "count": int(cnt),
                    "pct": round(cnt / total * 100, 2),
                }
            )

    return pd.DataFrame(rows)


def bridge_diagnostics(bridge: pd.DataFrame) -> dict:
    """
    Portfolio-size statistics from the landlord-company bridge.

    Parameters
    ----------
    bridge : pd.DataFrame with columns [LandLordID, CompanyID]

    Returns
    -------
    dict with keys: portfolio_size_stats, size_distribution, top_landlords
    """
    sizes = bridge.groupby("LandLordID")["CompanyID"].nunique().rename("n_companies")

    stats = {
        "portfolio_size_stats": sizes.describe().round(2).to_dict(),
        "size_distribution": sizes.value_counts().sort_index().to_dict(),
        "top_landlords": sizes.nlargest(10).to_dict(),
        "landlords_with_single_company": int((sizes == 1).sum()),
        "total_landlords": int(len(sizes)),
        "total_links": int(len(bridge)),
        "unique_companies": int(bridge["CompanyID"].nunique()),
    }

    logger.info(
        "Bridge diagnostics: %d landlords, %d companies, median portfolio size %.1f",
        stats["total_landlords"],
        stats["unique_companies"],
        sizes.median(),
    )
    return stats


# ---------------------------------------------------------------------------
# Plotting utilities (import matplotlib lazily to avoid hard dependency)
# ---------------------------------------------------------------------------


def _get_plt():
    """Lazy import of matplotlib.pyplot."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot functions. "
            "Install it with: pip install matplotlib"
        ) from exc


def plot_numeric_distributions(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    ncols: int = 4,
    figsize_per_cell: tuple = (4, 3),
    title: str = "Numeric Distributions",
) -> None:
    """
    Grid of histograms (with KDE) for numeric columns.

    Parameters
    ----------
    df              : DataFrame
    cols            : columns to plot; if None, all numeric columns (up to 32)
    ncols           : number of columns in the grid
    figsize_per_cell: (width, height) per subplot cell
    title           : figure suptitle
    """
    plt = _get_plt()
    import matplotlib.pyplot as mpl_plt

    num_cols = df.select_dtypes(include="number").columns.tolist() if cols is None else cols
    num_cols = [c for c in num_cols if c in df.columns][:32]  # cap at 32

    if not num_cols:
        logger.warning("No numeric columns to plot.")
        return

    nrows = (len(num_cols) + ncols - 1) // ncols
    fig_w = ncols * figsize_per_cell[0]
    fig_h = nrows * figsize_per_cell[1]
    fig, axes = mpl_plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    mpl_plt.tight_layout()
    mpl_plt.show()


def plot_category_frequencies(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    top_n: int = 10,
    ncols: int = 3,
    figsize_per_cell: tuple = (5, 3),
    title: str = "Category Frequencies",
) -> None:
    """
    Grid of horizontal bar charts for categorical columns.

    Parameters
    ----------
    df              : DataFrame
    cols            : columns to plot; if None, all object/category columns
    top_n           : max categories per chart
    ncols           : columns in grid
    figsize_per_cell: (width, height) per subplot
    title           : figure suptitle
    """
    plt = _get_plt()
    import matplotlib.pyplot as mpl_plt

    if cols is None:
        cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cols = [c for c in cols if c in df.columns]

    if not cols:
        logger.warning("No categorical columns to plot.")
        return

    nrows = (len(cols) + ncols - 1) // ncols
    fig_w = ncols * figsize_per_cell[0]
    fig_h = nrows * figsize_per_cell[1]
    fig, axes = mpl_plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        vc = df[col].value_counts(dropna=False).head(top_n)
        vc.sort_values().plot.barh(ax=ax, color="steelblue", alpha=0.8)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    mpl_plt.tight_layout()
    mpl_plt.show()


def plot_portfolio_distribution(bridge: pd.DataFrame) -> None:
    """
    Histogram of companies-per-landlord (portfolio size distribution).

    Parameters
    ----------
    bridge : pd.DataFrame with columns [LandLordID, CompanyID]
    """
    plt = _get_plt()
    import matplotlib.pyplot as mpl_plt

    sizes = bridge.groupby("LandLordID")["CompanyID"].nunique()

    fig, axes = mpl_plt.subplots(1, 2, figsize=(12, 4))

    # Linear scale
    axes[0].hist(sizes, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_title("Portfolio Size Distribution (linear)")
    axes[0].set_xlabel("Companies per Landlord")
    axes[0].set_ylabel("Landlord Count")
    axes[0].axvline(sizes.median(), color="red", linestyle="--", label=f"Median={sizes.median():.0f}")
    axes[0].legend()

    # Log scale
    axes[1].hist(sizes, bins=40, color="darkorange", edgecolor="white", alpha=0.8, log=True)
    axes[1].set_title("Portfolio Size Distribution (log y-scale)")
    axes[1].set_xlabel("Companies per Landlord")
    axes[1].set_ylabel("Landlord Count (log)")

    mpl_plt.suptitle("Landlord Portfolio Size", fontsize=13)
    mpl_plt.tight_layout()
    mpl_plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    figsize: tuple = (12, 10),
    title: str = "Pearson Correlation Heatmap",
) -> None:
    """
    Pearson correlation heatmap for numeric columns.

    Parameters
    ----------
    df     : DataFrame
    cols   : columns to include; if None, all numeric (up to 30)
    figsize: figure size
    title  : plot title
    """
    plt = _get_plt()
    import matplotlib.pyplot as mpl_plt

    num_df = df.select_dtypes(include="number")
    if cols is not None:
        num_df = num_df[[c for c in cols if c in num_df.columns]]
    num_df = num_df.iloc[:, :30]  # cap columns

    if num_df.empty or len(num_df.columns) < 2:
        logger.warning("Need at least 2 numeric columns for correlation heatmap.")
        return

    corr = num_df.corr()

    fig, ax = mpl_plt.subplots(figsize=figsize)
    try:
        import seaborn as sns
        sns.heatmap(
            corr,
            annot=len(corr) <= 15,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            ax=ax,
        )
    except ImportError:
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        mpl_plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)

    ax.set_title(title, fontsize=12)
    mpl_plt.tight_layout()
    mpl_plt.show()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _shape_info(df: pd.DataFrame, name: str) -> dict:
    return {
        "name": name,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": list(df.columns),
    }


def _id_integrity(df: pd.DataFrame, col: str, name: str) -> dict:
    if col not in df.columns:
        return {"dataset": name, "id_col": col, "present": False}
    return {
        "dataset": name,
        "id_col": col,
        "present": True,
        "rows": int(len(df)),
        "unique_ids": int(df[col].nunique(dropna=True)),
        "null_ids": int(df[col].isna().sum()),
        "duplicate_ids": int(len(df) - df[col].nunique(dropna=True)),
    }


def _numeric_records(summary: pd.DataFrame) -> list[dict]:
    if summary.empty:
        return []
    rows = []
    for col in summary.columns:
        row = {"column": col}
        for stat in summary.index:
            val = summary.loc[stat, col]
            row[str(stat)] = None if pd.isna(val) else float(val)
        rows.append(row)
    return rows


def _top_correlations(df: pd.DataFrame, n: int = 15) -> list[dict]:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return []
    corr = num.corr().abs()
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append({"a": a, "b": b, "abs_corr": round(float(v), 4)})
    return sorted(pairs, key=lambda x: -x["abs_corr"])[:n]


def _inspect_all_company_id_format(landlords: pd.DataFrame) -> dict:
    """Diagnose AllCompanyID storage format without changing parse behaviour."""
    if "AllCompanyID" not in landlords.columns:
        return {"present": False}

    sample = landlords["AllCompanyID"].dropna().head(20).tolist()
    types = sorted({type(v).__name__ for v in landlords["AllCompanyID"].dropna().head(500)})
    str_sample = [str(v)[:120] for v in sample[:5]]

    looks_like_python_list = 0
    looks_like_json = 0
    looks_like_csv = 0
    already_list = 0
    n = min(200, landlords["AllCompanyID"].notna().sum())
    for v in landlords["AllCompanyID"].dropna().head(n):
        if isinstance(v, (list, tuple)):
            already_list += 1
            continue
        raw = str(v).strip()
        if raw.startswith("[") and "'" in raw and '"' not in raw[:3]:
            looks_like_python_list += 1
        elif raw.startswith("["):
            looks_like_json += 1
        elif "," in raw:
            looks_like_csv += 1

    return {
        "present": True,
        "sample_types": types,
        "sample_values": str_sample,
        "of_first_n": n,
        "already_list": already_list,
        "looks_like_python_list_repr": looks_like_python_list,
        "looks_like_json": looks_like_json,
        "looks_like_csv": looks_like_csv,
    }


def _bridge_coverage(bridge: pd.DataFrame, companies: pd.DataFrame) -> dict:
    co_ids = set(companies["CompanyID"].astype(str))
    br_ids = set(bridge["CompanyID"].astype(str))
    unmatched = sorted(br_ids - co_ids)
    orphans = sorted(co_ids - br_ids)
    overlap = br_ids & co_ids

    malformed = [x for x in list(br_ids)[:5000] if "[" in x or "'" in x or '"' in x]
    # scan all for malformed rate
    n_malformed = sum(1 for x in br_ids if "[" in x or "'" in x)

    return {
        "companies_in_table": len(co_ids),
        "companies_in_bridge": len(br_ids),
        "overlap": len(overlap),
        "bridge_not_in_companies": len(unmatched),
        "companies_not_in_bridge": len(orphans),
        "match_rate_bridge_pct": round(len(overlap) / max(len(br_ids), 1) * 100, 2),
        "coverage_rate_companies_pct": round(len(overlap) / max(len(co_ids), 1) * 100, 2),
        "malformed_bridge_ids": n_malformed,
        "malformed_examples": unmatched[:8] if unmatched else list(br_ids)[:5],
        "orphan_examples": orphans[:8],
    }


def _collect_findings(
    landlords: pd.DataFrame,
    companies: pd.DataFrame,
    bridge: pd.DataFrame,
    ll_profile: pd.DataFrame,
    co_profile: pd.DataFrame,
    coverage: dict,
    fmt_info: dict,
    active_consistency: dict | None,
) -> list[dict]:
    """Structured findings for the report (issues + observations)."""
    findings: list[dict] = []

    malformed = coverage.get("malformed_bridge_ids", 0)
    python_list_n = fmt_info.get("looks_like_python_list_repr", 0)

    if malformed > 0:
        findings.append(
            {
                "severity": "CRITICAL",
                "area": "AllCompanyID parsing",
                "message": (
                    "AllCompanyID values look like Python list reprs with single quotes "
                    f"({python_list_n}/{fmt_info.get('of_first_n', '?')} sampled) and "
                    f"{malformed} bridge IDs still contain quotes/brackets after parsing."
                ),
                "detail": {
                    "samples": fmt_info.get("sample_values", []),
                    "malformed_examples": coverage.get("malformed_examples", []),
                },
            }
        )
    elif python_list_n > 0:
        findings.append(
            {
                "severity": "INFO",
                "area": "AllCompanyID parsing",
                "message": (
                    "AllCompanyID is stored as Python list reprs; auto-parser "
                    "(ast.literal_eval) handled them and bridge IDs are clean."
                ),
                "detail": {"samples": fmt_info.get("sample_values", [])},
            }
        )

    if coverage["match_rate_bridge_pct"] < 95:
        findings.append(
            {
                "severity": "CRITICAL" if coverage["match_rate_bridge_pct"] < 50 else "WARNING",
                "area": "Bridge coverage",
                "message": (
                    f"{coverage['match_rate_bridge_pct']}% of bridge CompanyIDs match "
                    f"Companies.parquet ({coverage['overlap']}/{coverage['companies_in_bridge']}). "
                    f"{coverage['bridge_not_in_companies']} unmatched in bridge; "
                    f"{coverage['companies_not_in_bridge']} companies never linked. "
                    "Unmatched IDs above the Companies table max are a data gap, not a parser bug."
                ),
                "detail": coverage,
            }
        )
    else:
        findings.append(
            {
                "severity": "INFO",
                "area": "Bridge coverage",
                "message": (
                    f"Bridge coverage is healthy: {coverage['match_rate_bridge_pct']}% of "
                    f"bridge IDs match Companies; {coverage['coverage_rate_companies_pct']}% of "
                    "companies appear in at least one portfolio."
                ),
                "detail": coverage,
            }
        )

    high_null_co = co_profile[co_profile["null_pct"] >= 40]
    if not high_null_co.empty:
        findings.append(
            {
                "severity": "WARNING",
                "area": "Companies missingness",
                "message": (
                    f"{len(high_null_co)} company columns have >=40% null "
                    f"({', '.join(high_null_co.index.tolist()[:8])}...)."
                ),
                "detail": high_null_co[["null_pct", "n_unique"]].to_dict("index"),
            }
        )

    high_null_ll = ll_profile[ll_profile["null_pct"] >= 20]
    if not high_null_ll.empty:
        findings.append(
            {
                "severity": "WARNING",
                "area": "Landlords missingness",
                "message": f"{len(high_null_ll)} landlord columns have >=20% null.",
                "detail": high_null_ll[["null_pct", "n_unique"]].to_dict("index"),
            }
        )

    if "CompanyStatus" in companies.columns:
        null_pct = float(companies["CompanyStatus"].isna().mean() * 100)
        none_pct = float((companies["CompanyStatus"].astype(str) == "None").mean() * 100)
        if null_pct + none_pct > 30:
            findings.append(
                {
                    "severity": "WARNING",
                    "area": "CompanyStatus label quality",
                    "message": (
                        f"CompanyStatus is sparse/ambiguous "
                        f"(null~={null_pct:.1f}%, literal 'None'~={none_pct:.1f}%). "
                        "Binary good/bad mapping will be noisy."
                    ),
                    "detail": {
                        "null_pct": round(null_pct, 2),
                        "none_pct": round(none_pct, 2),
                        "value_counts": companies["CompanyStatus"]
                        .value_counts(dropna=False)
                        .head(10)
                        .to_dict(),
                    },
                }
            )

    for col in ("SalesOfMainProduct", "SalesOfOtherProduct"):
        if col in companies.columns:
            zero_pct = float((companies[col] == 0).mean() * 100)
            if zero_pct >= 50:
                findings.append(
                    {
                        "severity": "INFO",
                        "area": "Zero-inflated sales",
                        "message": f"{col} is {zero_pct:.1f}% zeros - consider log1p / tree models.",
                        "detail": {"zero_pct": round(zero_pct, 2)},
                    }
                )

    if active_consistency and active_consistency.get("exact_match_pct", 100) < 95:
        findings.append(
            {
                "severity": "WARNING",
                "area": "ActiveCompanies consistency",
                "message": (
                    f"ActiveCompanies matches parsed AllCompanyID count for only "
                    f"{active_consistency['exact_match_pct']}% of landlords."
                ),
                "detail": active_consistency,
            }
        )
    elif active_consistency and active_consistency.get("exact_match_pct", 0) >= 95:
        findings.append(
            {
                "severity": "INFO",
                "area": "ActiveCompanies consistency",
                "message": (
                    f"ActiveCompanies aligns with parsed portfolio size for "
                    f"{active_consistency['exact_match_pct']}% of landlords."
                ),
                "detail": active_consistency,
            }
        )

    if landlords.duplicated().any() or companies.duplicated().any():
        findings.append(
            {
                "severity": "WARNING",
                "area": "Duplicate rows",
                "message": (
                    f"Duplicate full rows - landlords={int(landlords.duplicated().sum())}, "
                    f"companies={int(companies.duplicated().sum())}."
                ),
                "detail": {},
            }
        )
    else:
        findings.append(
            {
                "severity": "INFO",
                "area": "Row uniqueness",
                "message": "No full-row duplicates in Landlords or Companies.",
                "detail": {},
            }
        )

    return findings


def run_eda(
    landlords: pd.DataFrame | None = None,
    companies: pd.DataFrame | None = None,
    save_plots: bool = True,
    figures_dir: Path | str | None = None,
) -> dict:
    """
    Run full EDA on raw Landlords / Companies and return a serialisable report dict.

    Does not modify source data or fix quality issues — only measures and reports.

    Parameters
    ----------
    figures_dir : optional directory for plot export; defaults to reports/figures/eda
    """
    from src.config import cfg
    from src.data.ingestion import build_model_base, load_companies, load_landlords

    if landlords is None:
        landlords = load_landlords()
    if companies is None:
        companies = load_companies()

    bridge, _ = build_model_base(landlords, companies)

    ll_profile = profile_dataframe(landlords, "Landlords")
    co_profile = profile_dataframe(companies, "Companies")
    ll_num = numeric_summary(landlords)
    co_num = numeric_summary(companies)

    ll_cat_cols = [
        c
        for c in landlords.select_dtypes(include=["object", "category", "bool"]).columns
        if c != "AllCompanyID" and landlords[c].nunique(dropna=True) <= 60
    ]
    co_cat_cols = [
        c
        for c in companies.select_dtypes(include=["object", "category", "bool"]).columns
        if companies[c].nunique(dropna=True) <= 40
    ]
    ll_cats = category_summary(landlords, cols=ll_cat_cols, top_n=10)
    co_cats = category_summary(companies, cols=co_cat_cols, top_n=10)

    bridge_stats = bridge_diagnostics(bridge)
    fmt_info = _inspect_all_company_id_format(landlords)
    coverage = _bridge_coverage(bridge, companies)

    active_consistency = None
    if "ActiveCompanies" in landlords.columns:
        sizes = bridge.groupby("LandLordID")["CompanyID"].nunique()
        merged = landlords[["LandLordID", "ActiveCompanies"]].merge(
            sizes.rename("parsed_n").reset_index(), on="LandLordID", how="left"
        )
        merged["ActiveCompanies"] = pd.to_numeric(merged["ActiveCompanies"], errors="coerce")
        merged["diff"] = (merged["ActiveCompanies"] - merged["parsed_n"]).abs()
        active_consistency = {
            "exact_match_pct": round(float((merged["diff"] == 0).mean() * 100), 2),
            "within_1_pct": round(float((merged["diff"] <= 1).mean() * 100), 2),
            "median_diff": float(merged["diff"].median()) if len(merged) else None,
            "mean_claimed": float(merged["ActiveCompanies"].mean()),
            "mean_parsed": float(merged["parsed_n"].mean()),
        }

    sizes = bridge.groupby("LandLordID")["CompanyID"].nunique()
    bins = [1, 2, 3, 4, 5, 10, 20, 50, 100, 10**9]
    labels = ["1", "2", "3", "4", "5-9", "10-19", "20-49", "50-99", "100+"]
    bucketed = pd.cut(sizes, bins=bins, labels=labels, right=False)
    portfolio_buckets = (
        bucketed.value_counts().reindex(labels).fillna(0).astype(int).to_dict()
    )

    findings = _collect_findings(
        landlords,
        companies,
        bridge,
        ll_profile,
        co_profile,
        coverage,
        fmt_info,
        active_consistency,
    )

    figure_paths: list[str] = []
    if save_plots:
        target_figures = (
            Path(figures_dir)
            if figures_dir is not None
            else Path(cfg["paths"]["figures_dir"]) / "eda"
        )
        figure_paths = _save_eda_figures(
            landlords, companies, bridge, target_figures
        )

    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "shapes": [
            _shape_info(landlords, "Landlords"),
            _shape_info(companies, "Companies"),
        ],
        "id_integrity": [
            _id_integrity(landlords, "LandLordID", "Landlords"),
            _id_integrity(companies, "CompanyID", "Companies"),
        ],
        "all_company_id_format": fmt_info,
        "landlord_profile": ll_profile.reset_index().to_dict("records"),
        "company_profile": co_profile.reset_index().to_dict("records"),
        "landlord_numeric": _numeric_records(ll_num),
        "company_numeric": _numeric_records(co_num),
        "landlord_categories": ll_cats.to_dict("records") if not ll_cats.empty else [],
        "company_categories": co_cats.to_dict("records") if not co_cats.empty else [],
        "bridge": {
            **{
                k: (
                    {str(kk): vv for kk, vv in v.items()}
                    if isinstance(v, dict)
                    else v
                )
                for k, v in bridge_stats.items()
                if k != "size_distribution"
            },
            "portfolio_buckets": {str(k): int(v) for k, v in portfolio_buckets.items()},
            "coverage": coverage,
            "active_vs_parsed": active_consistency,
        },
        "correlations": {
            "landlord_top": _top_correlations(landlords),
            "company_top": _top_correlations(companies),
        },
        "findings": findings,
        "figures": figure_paths,
    }
    return report


def _save_eda_figures(
    landlords: pd.DataFrame,
    companies: pd.DataFrame,
    bridge: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    """Save key EDA plots to disk (no interactive show)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as mpl_plt
    except ImportError:
        logger.warning("matplotlib not available — skipping figure export")
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    # Portfolio size
    sizes = bridge.groupby("LandLordID")["CompanyID"].nunique()
    fig, axes = mpl_plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(sizes, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_title("Portfolio size (linear)")
    axes[0].set_xlabel("Companies per landlord")
    axes[0].set_ylabel("Landlord count")
    axes[0].axvline(sizes.median(), color="red", linestyle="--", label=f"median={sizes.median():.0f}")
    axes[0].legend()
    axes[1].hist(sizes, bins=40, color="darkorange", edgecolor="white", alpha=0.8, log=True)
    axes[1].set_title("Portfolio size (log y)")
    axes[1].set_xlabel("Companies per landlord")
    axes[1].set_ylabel("Landlord count (log)")
    fig.suptitle("Landlord portfolio size distribution")
    fig.tight_layout()
    p = figures_dir / "portfolio_size.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    mpl_plt.close(fig)
    paths.append(str(p))

    # Landlord numerics
    num_cols = landlords.select_dtypes(include="number").columns.tolist()
    if num_cols:
        ncols = min(4, len(num_cols))
        nrows = (len(num_cols) + ncols - 1) // ncols
        fig, axes = mpl_plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(num_cols):
            axes[i].hist(landlords[col].dropna(), bins=30, color="steelblue", edgecolor="white", alpha=0.8)
            axes[i].set_title(col, fontsize=9)
        for j in range(len(num_cols), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Landlords — numeric distributions")
        fig.tight_layout()
        p = figures_dir / "landlords_numeric.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        mpl_plt.close(fig)
        paths.append(str(p))

    # Company key numerics (cap)
    co_num_cols = companies.select_dtypes(include="number").columns.tolist()[:12]
    if co_num_cols:
        ncols = 4
        nrows = (len(co_num_cols) + ncols - 1) // ncols
        fig, axes = mpl_plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(co_num_cols):
            axes[i].hist(companies[col].dropna(), bins=30, color="teal", edgecolor="white", alpha=0.8)
            axes[i].set_title(col, fontsize=8)
        for j in range(len(co_num_cols), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Companies — numeric distributions (first 12)")
        fig.tight_layout()
        p = figures_dir / "companies_numeric.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        mpl_plt.close(fig)
        paths.append(str(p))

    # Category bars
    for name, df, cols, fname in [
        (
            "Landlords",
            landlords,
            [c for c in ["PreferredIndustry", "LandlordOriginCountry"] if c in landlords.columns],
            "landlords_categories.png",
        ),
        (
            "Companies",
            companies,
            [c for c in ["CompanyStatus", "PrimaryType"] if c in companies.columns],
            "companies_categories.png",
        ),
    ]:
        if not cols:
            continue
        fig, axes = mpl_plt.subplots(1, len(cols), figsize=(6 * len(cols), 4))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            vc = df[col].value_counts(dropna=False).head(12).sort_values()
            vc.plot.barh(ax=ax, color="steelblue", alpha=0.85)
            ax.set_title(f"{name}: {col}")
        fig.tight_layout()
        p = figures_dir / fname
        fig.savefig(p, dpi=120, bbox_inches="tight")
        mpl_plt.close(fig)
        paths.append(str(p))

    # Correlation heatmap (companies)
    num = companies.select_dtypes(include="number").iloc[:, :20]
    if num.shape[1] >= 2:
        corr = num.corr()
        fig, ax = mpl_plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)
        ax.set_title("Companies — Pearson correlation (first 20 numeric)")
        fig.tight_layout()
        p = figures_dir / "companies_correlation.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        mpl_plt.close(fig)
        paths.append(str(p))

    logger.info("Saved %d EDA figures to %s", len(paths), figures_dir)
    return paths


def _df_to_md_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "_No rows._\n"
    cols = columns or list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def render_eda_markdown(report: dict) -> str:
    """Render the EDA report dict as a Markdown document."""
    lines: list[str] = []
    lines.append("# Exploratory Data Analysis Report")
    lines.append("")
    lines.append(f"_Generated at: {report.get('generated_at', '')}_")
    lines.append("")
    lines.append("Raw inputs: `data/raw/Landlords.parquet`, `data/raw/Companies.parquet`")
    lines.append("")

    # Findings first
    lines.append("## Findings (review before fixes)")
    lines.append("")
    findings = report.get("findings", [])
    if not findings:
        lines.append("No structured findings.")
    else:
        for i, f in enumerate(findings, 1):
            lines.append(f"### {i}. [{f['severity']}] {f['area']}")
            lines.append("")
            lines.append(f['message'])
            lines.append("")
            if f.get("detail"):
                lines.append("```json")
                lines.append(json.dumps(f["detail"], indent=2, default=str)[:3000])
                lines.append("```")
                lines.append("")

    lines.append("## Dataset shapes")
    lines.append("")
    lines.append(
        _df_to_md_table(
            [
                {
                    "dataset": s["name"],
                    "rows": s["rows"],
                    "cols": s["cols"],
                    "memory_mb": s["memory_mb"],
                    "duplicate_rows": s["duplicate_rows"],
                }
                for s in report["shapes"]
            ]
        )
    )

    lines.append("## Identifier integrity")
    lines.append("")
    lines.append(_df_to_md_table(report["id_integrity"]))

    lines.append("## AllCompanyID format inspection")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report.get("all_company_id_format", {}), indent=2, default=str))
    lines.append("```")
    lines.append("")

    lines.append("## Column profiles — Landlords")
    lines.append("")
    lines.append(_df_to_md_table(report["landlord_profile"]))

    lines.append("## Column profiles — Companies")
    lines.append("")
    lines.append(_df_to_md_table(report["company_profile"]))

    lines.append("## Numeric summary — Landlords")
    lines.append("")
    lines.append(_df_to_md_table(report["landlord_numeric"]))

    lines.append("## Numeric summary — Companies")
    lines.append("")
    lines.append(_df_to_md_table(report["company_numeric"]))

    lines.append("## Category frequencies — Landlords")
    lines.append("")
    lines.append(_df_to_md_table(report["landlord_categories"]))

    lines.append("## Category frequencies — Companies")
    lines.append("")
    lines.append(_df_to_md_table(report["company_categories"]))

    lines.append("## Bridge & portfolio")
    lines.append("")
    bridge = report.get("bridge", {})
    lines.append("```json")
    lines.append(
        json.dumps(
            {k: v for k, v in bridge.items() if k != "top_landlords"},
            indent=2,
            default=str,
        )
    )
    lines.append("```")
    lines.append("")
    if bridge.get("top_landlords"):
        lines.append("### Top landlords by portfolio size")
        lines.append("")
        lines.append(
            _df_to_md_table(
                [{"LandLordID": k, "n_companies": v} for k, v in bridge["top_landlords"].items()]
            )
        )

    lines.append("## Top correlations")
    lines.append("")
    lines.append("### Landlords")
    lines.append(_df_to_md_table(report.get("correlations", {}).get("landlord_top", [])))
    lines.append("### Companies")
    lines.append(_df_to_md_table(report.get("correlations", {}).get("company_top", [])))

    if report.get("figures"):
        lines.append("## Figures")
        lines.append("")
        for fig in report["figures"]:
            rel = fig
            lines.append(f"- `{rel}`")
            # Embed relative path if under reports/
            lines.append("")
            lines.append(f"![figure]({Path(fig).as_posix()})")
            lines.append("")

    lines.append("---")
    lines.append("_Issues listed above are intentional for review; do not treat this report as a fix log._")
    lines.append("")
    return "\n".join(lines)


def write_eda_report(
    report: dict | None = None,
    *,
    save_plots: bool = True,
) -> tuple[Path, Path]:
    """
    Run EDA (if needed) and write JSON + Markdown under reports/eda_runs/<timestamp>/.

    Previous runs are never overwritten. Convenience copies of the newest report
    are also written to reports/eda_report.{json,md} (latest only).

    Returns
    -------
    (json_path, markdown_path)  — paths inside the timestamped run directory
    """
    from datetime import datetime

    from src.config import cfg

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = reports_dir / "eda_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Preserve any pre-versioning flat reports once, then keep appending runs
    _archive_legacy_flat_reports(reports_dir, runs_dir)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / stamp
    # Avoid collision if re-run in the same second
    if run_dir.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = runs_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    run_figures_dir = run_dir / "figures"

    # Point figure export at this run's folder
    if report is None:
        report = run_eda(save_plots=save_plots, figures_dir=run_figures_dir)

    report = dict(report)
    report["run_id"] = stamp
    report["run_dir"] = str(run_dir)

    json_path = run_dir / "eda_report.json"
    md_path = run_dir / "eda_report.md"

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        if pd.isna(o):
            return None
        return str(o)

    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, default=_default)

    md = render_eda_markdown(report)
    # Prefer paths relative to project root for portability in markdown
    root = reports_dir.resolve().parent
    for fig in report.get("figures", []):
        try:
            rel = Path(fig).resolve().relative_to(root)
            md = md.replace(Path(fig).as_posix(), rel.as_posix())
            md = md.replace(str(fig), rel.as_posix())
        except ValueError:
            pass

    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write(md)

    # Refresh "latest" convenience copies without deleting historical runs
    latest_json = reports_dir / "eda_report.json"
    latest_md = reports_dir / "eda_report.md"
    with open(latest_json, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, default=_default)
    with open(latest_md, "w", encoding="utf-8") as fp:
        fp.write(md)

    logger.info(
        "EDA report archived to %s (latest copies: %s, %s)",
        run_dir,
        latest_json,
        latest_md,
    )
    return json_path, md_path


def _archive_legacy_flat_reports(reports_dir: Path, runs_dir: Path) -> None:
    """
    One-time migration: if flat eda_report.* exist and no runs yet (or they are
    not already mirrored), copy them into eda_runs/legacy_<mtime>/ so history
    is not lost when latest copies are refreshed.
    """
    import shutil
    from datetime import datetime

    flat_json = reports_dir / "eda_report.json"
    flat_md = reports_dir / "eda_report.md"
    if not flat_json.exists() and not flat_md.exists():
        return

    # Skip if we already have at least one run and flat files look like "latest"
    # copies of the newest run (same size/content check via mtime after first run).
    # Always archive flat files that predate any run directory.
    existing_runs = [p for p in runs_dir.iterdir() if p.is_dir()] if runs_dir.exists() else []
    if existing_runs:
        return

    mtime = None
    for p in (flat_json, flat_md):
        if p.exists():
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            break
    stamp = mtime.strftime("%Y%m%d_%H%M%S") if mtime else "legacy"
    legacy_dir = runs_dir / f"{stamp}_legacy"
    if legacy_dir.exists():
        return
    legacy_dir.mkdir(parents=True, exist_ok=True)

    for src, name in ((flat_json, "eda_report.json"), (flat_md, "eda_report.md")):
        if src.exists():
            shutil.copy2(src, legacy_dir / name)

    # Also preserve any existing shared figures folder
    old_figures = Path(reports_dir) / "figures" / "eda"
    if old_figures.exists() and any(old_figures.iterdir()):
        dest_fig = legacy_dir / "figures"
        shutil.copytree(old_figures, dest_fig, dirs_exist_ok=True)

    logger.info("Archived previous flat EDA report to %s", legacy_dir)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    json_path, md_path = write_eda_report(save_plots=True)
    # Reload for summary print
    with open(json_path, encoding="utf-8") as fp:
        _report = json.load(fp)

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print(f"Run dir:  {json_path.parent}")
    print(f"JSON:     {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Latest:   reports/eda_report.json | reports/eda_report.md")
    print(f"Figures:  {len(_report.get('figures', []))} files")
    print(f"Findings: {len(_report.get('findings', []))}")
    print("-" * 60)
    for i, f in enumerate(_report.get("findings", []), 1):
        msg = (
            f["message"][:100]
            .replace("≥", ">=")
            .replace("–", "-")
            .replace("—", "-")
            .replace("≈", "~=")
            .replace("…", "...")
        )
        print(f"  {i}. [{f['severity']}] {f['area']}: {msg}")
    print("=" * 60)
