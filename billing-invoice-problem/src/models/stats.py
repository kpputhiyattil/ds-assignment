"""Significance testing for the ablation (Step 6).

* **Paired bootstrap** -- the general-purpose test for PR-AUC, dollar-recall and
  monetary-error deltas, where two models are scored on the *same* customers so
  their errors are correlated. Resampling customers (with replacement) and
  recomputing the metric difference gives a distribution of the delta, hence a
  CI and a two-sided p-value.
* **DeLong** -- the exact, closed-form test for the difference of two correlated
  ROC-AUCs (Sun & Xu 2014 fast algorithm). Used for ROC-AUC only, as designed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def paired_bootstrap_delta(
    compute: Callable[[np.ndarray], tuple[float, float]],
    n: int,
    n_boot: int = 300,
    seed: int = 42,
) -> dict:
    """Bootstrap the delta (treatment - base) of a metric over paired samples.

    ``compute(idx)`` returns ``(base_value, treat_value)`` on the resampled
    indices. Returns the observed delta, 95% CI, and a two-sided p-value for
    ``delta != 0``.
    """
    rng = np.random.default_rng(seed)
    base_full, treat_full = compute(np.arange(n))
    observed = treat_full - base_full

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        b, t = compute(idx)
        deltas[i] = t - b

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_two_sided = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "base": round(float(base_full), 4),
        "treatment": round(float(treat_full), 4),
        "delta": round(float(observed), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_value": round(float(min(p_two_sided, 1.0)), 4),
        "significant": bool(lo > 0 or hi < 0),
    }


# ---------------------------------------------------------------------------
# Fast DeLong test for two correlated ROC-AUCs (Sun & Xu 2014).
# ---------------------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """preds_sorted: (k, n) with the m positive samples first. Returns aucs, cov."""
    k, n = preds_sorted.shape
    npos, nneg = m, n - m
    tx = np.empty((k, npos))
    ty = np.empty((k, nneg))
    tz = np.empty((k, n))
    for r in range(k):
        tx[r] = _midrank(preds_sorted[r, :npos])
        ty[r] = _midrank(preds_sorted[r, npos:])
        tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :npos].sum(axis=1) / npos / nneg - (npos + 1.0) / 2.0 / nneg
    v01 = (tz[:, :npos] - tx) / nneg
    v10 = 1.0 - (tz[:, npos:] - ty) / npos
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / npos + sy / nneg
    return aucs, np.atleast_2d(cov)


def delong_roc_test(y_true: np.ndarray, p_base: np.ndarray, p_treat: np.ndarray) -> dict:
    """DeLong test for AUC(treat) - AUC(base) on the same samples."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)  # positives (1) first
    m = int(y_true.sum())
    preds = np.vstack([np.asarray(p_base)[order], np.asarray(p_treat)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    delta = aucs[1] - aucs[0]
    if var <= 0:
        z, p = 0.0, 1.0
    else:
        z = delta / np.sqrt(var)
        p = 2.0 * stats.norm.sf(abs(z))
    return {
        "auc_base": round(float(aucs[0]), 4),
        "auc_treatment": round(float(aucs[1]), 4),
        "delta": round(float(delta), 4),
        "z": round(float(z), 4),
        "p_value": round(float(p), 4),
        "significant": bool(p < 0.05 and delta > 0),
    }
