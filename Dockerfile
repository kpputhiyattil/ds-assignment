FROM python:3.11-slim

WORKDIR /app

# Install system deps for CatBoost and pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Default: run the full pipeline
CMD ["python", "-m", "src.data.ingestion"]
