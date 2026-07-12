"""Human-readable explanations for interval inference and credit decisions."""

from __future__ import annotations

from typing import Any

INTERVAL_PLAIN = {
    "insufficient_history": "not enough invoice history yet to tell the billing cycle",
    "one_time_candidate": "looks like a one-time / non-recurring charge pattern",
    "monthly": "monthly billing (about every month)",
    "quarterly": "quarterly billing (about every 3 months)",
    "semi_annual": "semi-annual billing (about every 6 months)",
    "annual": "annual billing (about once a year)",
    "mixed": "mixed cadence (more than one regular billing pattern)",
    "irregular": "irregular / no clear calendar billing cycle",
}


def explain_inferred_interval(row: dict[str, Any]) -> str:
    """Plain-language reason the rule engine assigned ``billing_interval``."""
    label = row.get("inferred_interval") or row.get("billing_interval") or "unknown"
    plain = INTERVAL_PLAIN.get(str(label), str(label).replace("_", " "))
    conf = float(row.get("interval_confidence") or 0.0)
    n_events = int(row.get("n_events") or 0)
    n_gaps = int(row.get("n_gaps") or 0)
    med = row.get("median_gap_days")
    gap_cv = row.get("gap_cv")
    dominant = row.get("dominant_base")
    share = row.get("dominant_share")
    n_bases = row.get("n_distinct_bases")
    recency = row.get("recency_days")

    parts: list[str] = [f"We inferred **{plain}**"]

    if label == "insufficient_history":
        if n_events <= 1:
            parts.append(
                f"because only {n_events} billing day(s) were observed before the "
                "assessment date — that is too little history to confirm a cycle"
            )
        else:
            parts.append(
                "because the recent history is still too short to confirm a stable cycle"
            )
        if recency is not None:
            parts.append(f"(last invoice about {int(recency)} days before the snapshot)")
    elif label == "one_time_candidate":
        parts.append(
            "because there is essentially a single invoice followed by a long quiet "
            "period with no repeat billing"
        )
    elif label == "irregular":
        med_s = f"{float(med):.0f} days" if med is not None else "unknown"
        parts.append(
            f"because the gaps between invoices (typical gap ≈ {med_s}) do not line up "
            "cleanly with monthly / quarterly / semi-annual / annual calendars"
        )
    elif label == "mixed":
        parts.append(
            "because invoice gaps cluster around more than one calendar period "
            "(for example a monthly subscription plus an annual fee)"
        )
        if dominant and share is not None:
            parts.append(
                f"— the strongest pattern is {dominant} "
                f"(about {100 * float(share):.0f}% of matched gaps)"
            )
    else:
        med_s = f"{float(med):.0f} days" if med is not None else "n/a"
        parts.append(
            f"because the typical gap between billing days is about {med_s}, which "
            f"matches {plain}"
        )
        if n_gaps:
            parts.append(f"across {n_gaps} observed cycle(s)")
        if dominant and share is not None:
            parts.append(
                f"(about {100 * float(share):.0f}% of gaps fit that pattern"
                + (f"; {int(n_bases)} competing patterns seen" if n_bases and int(n_bases) > 1 else "")
                + ")"
            )
        if gap_cv is not None and float(gap_cv) == float(gap_cv):
            consistency = (
                "very regular" if float(gap_cv) < 0.2
                else "fairly regular" if float(gap_cv) < 0.5
                else "somewhat uneven"
            )
            parts.append(f"and spacing looks {consistency}")

    parts.append(f"Confidence in this label is **{conf:.0%}**.")
    return " ".join(parts)


def explain_recommendation(
    *,
    recommendation: str,
    reason: str,
    detail: str,
    churn_probability: float | None,
    interval_confidence: float | None,
    inferred_interval: str | None = None,
    model_tag: str = "with interval features",
) -> str:
    """Plain-language reason for Continue / Review / No-Go."""
    p = churn_probability
    conf = interval_confidence
    interval_plain = INTERVAL_PLAIN.get(
        str(inferred_interval or ""), str(inferred_interval or "unknown cycle")
    )

    if reason == "no_history":
        return (
            "We recommended **Review** because there is no usable invoice history "
            "before the assessment date, so the model cannot score this customer yet."
        )

    if reason == "low_interval_confidence":
        return (
            f"We recommended **Review** ({model_tag}) because the billing-cycle "
            f"signal is too uncertain "
            f"(confidence {conf:.0%} for “{interval_plain}”). "
            "Thin or ambiguous history is treated as a data-quality issue — "
            "we do not auto Continue or auto No-Go until the cycle is clearer."
        )

    if p is None:
        return f"We recommended **{recommendation}**. {detail}"

    risk_words = (
        "low" if p <= 0.3 else "elevated" if p < 0.7 else "high"
    )
    base = (
        f"Using the model **{model_tag}**, predicted chance of dollar churn is "
        f"**{p:.0%}** ({risk_words} risk)"
    )

    if recommendation == "Continue":
        return (
            f"{base}. That sits below the Continue threshold, and interval "
            f"confidence ({conf:.0%}) is high enough to auto-decide, so the "
            f"recommendation is **Continue** — proceed with onboarding."
        )
    if recommendation == "No-Go":
        return (
            f"{base}. That sits above the No-Go threshold, and interval "
            f"confidence ({conf:.0%}) is high enough to auto-decide, so the "
            f"recommendation is **No-Go** — decline or escalate for credit risk."
        )
    # borderline Review
    return (
        f"{base}. That sits between the Continue and No-Go bands "
        f"(or still needs analyst judgment), so the recommendation is **Review**."
    )


def explain_model_drivers(briefing: dict[str, Any] | None) -> str:
    """Turn SHAP briefing into short human sentences (not raw feature dumps)."""
    if not briefing:
        return (
            "Detailed factor-level explanation was not computed for this row "
            "(enable explain_top_n / select the customer to expand drivers)."
        )

    bits: list[str] = []
    tier = briefing.get("tier")
    if tier:
        bits.append(f"Overall risk tier: **{tier}**.")

    main = briefing.get("main_reason") or {}
    if main.get("label"):
        direction = "raising" if main.get("direction") == "increasing" else "lowering"
        bits.append(
            f"The single biggest driver is **{main['label']}**, which is currently "
            f"**{direction}** churn risk."
        )

    ups = briefing.get("risk_increasing") or []
    downs = briefing.get("risk_decreasing") or []
    if ups:
        bits.append(
            "Factors pushing risk up: "
            + "; ".join(
                f"{u.get('label')} (value {u.get('value')})" for u in ups[:3]
            )
            + "."
        )
    if downs:
        bits.append(
            "Factors pulling risk down: "
            + "; ".join(
                f"{d.get('label')} (value {d.get('value')})" for d in downs[:3]
            )
            + "."
        )
    return " ".join(bits)


def compare_decisions(with_reco: str, without_reco: str, p_with: float, p_wo: float) -> str:
    """Explain how adding the inferred interval changed the outcome."""
    delta = p_with - p_wo
    if with_reco == without_reco:
        return (
            f"Both models agree on **{with_reco}**. "
            f"Adding the inferred interval changed churn probability by "
            f"{delta:+.1%} (with={p_with:.0%}, without={p_wo:.0%})."
        )
    return (
        f"The models **disagree**: with interval → **{with_reco}**, "
        f"without interval → **{without_reco}**. "
        f"Churn probability shifts from {p_wo:.0%} (no interval) to "
        f"{p_with:.0%} (with interval), Δ={delta:+.1%}. "
        "This is exactly the decision-relevant value of the inferred cycle."
    )
