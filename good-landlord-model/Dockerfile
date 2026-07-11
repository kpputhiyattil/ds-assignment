FROM python:3.11-slim

WORKDIR /app

# System deps for CatBoost / LightGBM / pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

# Default: unit tests (no raw data required).
# Override for pipeline, e.g.:
#   docker run ... python scripts/run_landlord_models.py
CMD ["pytest", "tests/", "-q"]
