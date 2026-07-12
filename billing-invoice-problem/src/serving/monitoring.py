"""Production monitoring: feature/prediction drift (Step 8).

Population Stability Index (PSI) compares a live feature (or prediction)
distribution against the training reference. Rule of thumb: PSI < 0.1 = stable,
0.1-0.25 = moderate shift (investigate), > 0.25 = significant drift (retrain
trigger). Bin edges are fixed from the reference so the same buckets are used
over time.
"""

from __future__ import annotations

import numpy as np

PSI_STABLE = 0.1
PSI_SIGNIFICANT = 0.25


def psi(reference: np.ndarray, actual: np.ndarray, n_bins: int = 10,
        eps: float = 1e-6) -> float:
    """Population Stability Index between a reference and an actual sample."""
    reference = np.asarray(reference, dtype=float)
    reference = reference[~np.isnan(reference)]
    actual = np.asarray(actual, dtype=float)
    actual = actual[~np.isnan(actual)]
    if reference.size == 0 or actual.size == 0:
        return float("nan")

    # Quantile bin edges from the reference (robust to skew); dedupe for ties.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / reference.size
    act_pct = np.histogram(actual, bins=edges)[0] / actual.size
    ref_pct = np.clip(ref_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)
    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def drift_label(psi_value: float) -> str:
    if np.isnan(psi_value):
        return "unknown"
    if psi_value < PSI_STABLE:
        return "stable"
    if psi_value < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def population_stability_report(
    reference: dict[str, np.ndarray], actual: dict[str, np.ndarray]
) -> dict:
    """Per-feature PSI + drift label for every shared feature."""
    report = {}
    for feat in reference:
        if feat in actual:
            val = psi(reference[feat], actual[feat])
            report[feat] = {"psi": round(val, 4), "drift": drift_label(val)}
    n_sig = sum(1 for r in report.values() if r["drift"] == "significant")
    return {"features": report, "n_significant_drift": n_sig,
            "action": "retrain" if n_sig else "none"}
