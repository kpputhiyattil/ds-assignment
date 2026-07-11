"""
ingestion.py — Load raw parquet files and build the landlord-company bridge table.

Responsibilities
----------------
1. Load LandLords.parquet and Companies.parquet from configured paths.
2. Parse AllCompanyID (handles list, JSON-string, or comma-separated string formats).
3. Build a normalized bridge table: one row per (LandLordID, CompanyID) relationship.
4. Merge bridge with Companies to produce model_base — the row-level unit for
   company baseline modelling.

This module deliberately contains NO feature engineering logic.
Feature transforms live in src/features/transforms.py so they can be shared
between training and any future serving layer.
"""
from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_landlords(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load LandLords.parquet. Returns raw DataFrame unchanged."""
    fpath = Path(path or cfg["paths"]["raw_landlords"])
    if not fpath.exists():
        raise FileNotFoundError(
            f"LandLords file not found: {fpath}\n"
            "Place LandLords.parquet in data/raw/ before running the pipeline."
        )
    df = pd.read_parquet(fpath)
    logger.info("Loaded LandLords: %d rows × %d cols", *df.shape)
    return df


def load_companies(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load Companies.parquet. Returns raw DataFrame unchanged."""
    fpath = Path(path or cfg["paths"]["raw_companies"])
    if not fpath.exists():
        raise FileNotFoundError(
            f"Companies file not found: {fpath}\n"
            "Place Companies.parquet in data/raw/ before running the pipeline."
        )
    df = pd.read_parquet(fpath)
    logger.info("Loaded Companies: %d rows × %d cols", *df.shape)
    return df


def parse_all_company_id(value, fmt: str = "auto") -> list[str]:
    """
    Parse a single AllCompanyID cell into a list of company ID strings.

    Supported formats (fmt):
        "list"         — value is already a Python list
        "json_string"  — value is a JSON array string: '["id1","id2"]'
        "comma_string" — value is a plain comma-separated string: "id1,id2"
        "auto"         — tries list → json → comma in order

    Returns an empty list for null/empty values.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if fmt == "list" or (fmt == "auto" and isinstance(value, list)):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        if fmt == "json_string" or fmt == "auto":
            # Try JSON first
            if value.startswith("["):
                try:
                    parsed = json.loads(value)
                    return [str(v).strip() for v in parsed if str(v).strip()]
                except (json.JSONDecodeError, ValueError):
                    pass
                # Fallback: ast.literal_eval for Python-style lists
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, (list, tuple)):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except (ValueError, SyntaxError):
                    pass

        # Comma-separated fallback
        return [v.strip() for v in value.split(",") if v.strip()]

    # Scalar value (single ID)
    return [str(value).strip()]


def build_landlord_company_bridge(
    landlords: pd.DataFrame,
    fmt: Optional[str] = None,
) -> pd.DataFrame:
    """
    Explode AllCompanyID into a normalized bridge table.

    Returns DataFrame with columns: [LandLordID, CompanyID]
    One row per unique (LandLordID, CompanyID) pair.
    """
    fmt = fmt or cfg["data"]["all_company_id_format"]

    if "AllCompanyID" not in landlords.columns:
        raise KeyError(
            "'AllCompanyID' column not found in LandLords dataset. "
            "Check the raw file schema."
        )
    if "LandLordID" not in landlords.columns:
        raise KeyError("'LandLordID' column not found in LandLords dataset.")

    bridge = (
        landlords[["LandLordID", "AllCompanyID"]]
        .copy()
        .assign(
            CompanyID=lambda df: df["AllCompanyID"].apply(
                lambda v: parse_all_company_id(v, fmt=fmt)
            )
        )
        .explode("CompanyID")
        .drop(columns=["AllCompanyID"])
        .dropna(subset=["CompanyID"])
    )

    # Normalise types — ensure both IDs are strings for consistent merging
    bridge["LandLordID"] = bridge["LandLordID"].astype(str).str.strip()
    bridge["CompanyID"] = bridge["CompanyID"].astype(str).str.strip()

    # Remove blanks introduced by parsing
    bridge = bridge[bridge["CompanyID"] != ""]
    bridge = bridge.drop_duplicates()

    logger.info(
        "Bridge table: %d (LandLordID, CompanyID) pairs from %d unique landlords",
        len(bridge),
        bridge["LandLordID"].nunique(),
    )
    return bridge.reset_index(drop=True)


def build_model_base(
    landlords: pd.DataFrame,
    companies: pd.DataFrame,
    fmt: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the main modelling table by joining bridge → companies.

    Returns
    -------
    bridge : pd.DataFrame
        Normalized (LandLordID, CompanyID) pairs.
    model_base : pd.DataFrame
        Bridge merged with company features; one row per tenant-landlord pair.
        Companies not found in Companies.parquet appear with NaN company columns.
    """
    bridge = build_landlord_company_bridge(landlords, fmt=fmt)

    companies = companies.copy()
    companies["CompanyID"] = companies["CompanyID"].astype(str).str.strip()

    model_base = bridge.merge(companies, on="CompanyID", how="left")

    unmatched = model_base["CompanyID"].isna().sum()
    if unmatched > 0:
        logger.warning(
            "%d bridge rows could not be matched to a company record. "
            "These will have NaN for all company columns.",
            unmatched,
        )

    matched_pct = (model_base["CompanyID"].notna().sum() / len(model_base)) * 100
    logger.info(
        "model_base: %d rows (%.1f%% matched to company records)",
        len(model_base),
        matched_pct,
    )
    return bridge, model_base


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Data Ingestion ===")

    landlords = load_landlords()
    companies = load_companies()

    bridge, model_base = build_model_base(landlords, companies)

    # Save intermediate outputs for downstream steps
    bridge_path = processed_dir / "landlord_company_bridge.parquet"
    base_path = processed_dir / "model_base.parquet"

    bridge.to_parquet(bridge_path, index=False)
    model_base.to_parquet(base_path, index=False)

    logger.info("Saved bridge → %s", bridge_path)
    logger.info("Saved model_base → %s", base_path)

    print("\nIngestion complete.")
    print(f"  LandLords:  {landlords.shape}")
    print(f"  Companies:  {companies.shape}")
    print(f"  Bridge:     {bridge.shape}")
    print(f"  Model base: {model_base.shape}")
    print(f"\nOutputs saved to {processed_dir}")
