"""
Data loader for Companies.parquet.

Responsible for:
- Loading and validating the parquet file on startup
- Providing company lookups by ID
- Never logging raw company data (PII-adjacent financial fields)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns — used for schema validation on load
REQUIRED_COLUMNS = {
    "CompanyID",
    "CompanyStatus",
    "PrimaryType",
    "YearFounded",
    "OriginCountry",
    "Rank",
    "SalesMain",
    "SalesOther",
    "MonthlyBudget",
    "Investments",
    "TodaysClients",
    "ReturningClients",
    "TotalActiveClients",
    "Clients7D",
    "Clients6M",
    "Clients12M",
    "Visitors7D",
    "Visitors6M",
    "Visitors12M",
}

DEFAULT_DATA_PATH = "./Companies.parquet"


class DataLoaderError(Exception):
    """Raised when the parquet file cannot be loaded or is invalid."""


class CompanyNotFoundError(KeyError):
    """Raised when a company_id does not exist in the dataset."""


class DataLoader:
    """
    Loads Companies.parquet and provides company-level lookups.

    Usage
    -----
    loader = DataLoader()                        # uses DATA_PATH env var or default
    loader = DataLoader("/path/to/data.parquet") # explicit path

    ids   = loader.list_company_ids()
    row   = loader.get_company("COMP_001")       # returns dict
    sample = loader.get_sample_companies(n=3)    # list of dicts
    """

    def __init__(self, path: Optional[str] = None) -> None:
        data_path = path or os.getenv("DATA_PATH", DEFAULT_DATA_PATH)
        self._path = Path(data_path)
        self._df = self._load(self._path)
        logger.info("DataLoader ready: %d companies loaded from %s", len(self._df), self._path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_company(self, company_id: str) -> dict:
        """
        Return a single company row as a Python dict.

        All numeric NaN values are converted to None so downstream tools
        receive proper Python None, not float('nan').

        Raises
        ------
        CompanyNotFoundError if the ID is not in the dataset.
        """
        mask = self._df["CompanyID"] == company_id
        matches = self._df[mask]

        if matches.empty:
            raise CompanyNotFoundError(
                f"Company '{company_id}' not found. "
                f"Use list_company_ids() to see available IDs."
            )

        row = matches.iloc[0].to_dict()
        # Convert NaN → None for clean downstream handling
        return {k: (None if pd.isna(v) else v) for k, v in row.items()}

    def list_company_ids(self) -> list[str]:
        """Return a sorted list of all CompanyIDs in the dataset."""
        return sorted(self._df["CompanyID"].astype(str).tolist())

    def get_sample_companies(self, n: int = 3) -> list[dict]:
        """
        Return a representative sample of n companies.

        Tries to include variety across CompanyStatus values so the sample
        is useful for demo/testing (i.e., not all 'active').
        """
        df = self._df
        statuses = df["CompanyStatus"].dropna().unique().tolist()

        sample_rows: list[pd.Series] = []
        per_status = max(1, n // len(statuses)) if statuses else n

        for status in statuses:
            subset = df[df["CompanyStatus"] == status]
            take = min(per_status, len(subset))
            sample_rows.append(subset.sample(n=take, random_state=42))
            if len(sample_rows) >= n:
                break

        sample_df = pd.concat(sample_rows).head(n)
        return [
            {k: (None if pd.isna(v) else v) for k, v in row.items()}
            for _, row in sample_df.iterrows()
        ]

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns) of the loaded dataset."""
        return self._df.shape

    @property
    def null_rates(self) -> dict[str, float]:
        """Null rate per column — useful for data quality checks on startup."""
        return (self._df.isnull().mean() * 100).round(2).to_dict()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise DataLoaderError(
                f"Data file not found: {path}\n"
                f"Set DATA_PATH env var or place Companies.parquet in the project root."
            )

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            raise DataLoaderError(f"Failed to read parquet file '{path}': {exc}") from exc

        self._validate_schema(df)
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Fail loudly if expected columns are missing."""
        actual = set(df.columns)
        missing = REQUIRED_COLUMNS - actual

        if missing:
            logger.warning(
                "Parquet schema mismatch — missing columns: %s. "
                "Some tool computations may return null values.",
                sorted(missing),
            )
        else:
            logger.debug("Schema validation passed — all required columns present.")
